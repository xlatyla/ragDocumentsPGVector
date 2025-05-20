import os
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import Text, select
from sqlalchemy.dialects.postgresql import JSONB
from pgvector.sqlalchemy import Vector
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
import aiohttp
from typing import List, Dict, Any
import json
from PyPDF2 import PdfReader

def extract_text(doc_path: str) -> str:
    """
    Extrae el texto de un archivo PDF.
    
    Args:
        doc_path (str): Ruta al archivo PDF
        
    Returns:
        str: Texto extraído del PDF
        
    Raises:
        FileNotFoundError: Si el archivo no existe
        Exception: Si hay un error al procesar el PDF
    """
    try:
        reader = PdfReader(doc_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except FileNotFoundError:
        raise FileNotFoundError(f"El archivo {doc_path} no existe")
    except Exception as e:
        raise Exception(f"Error al procesar el PDF: {str(e)}")

class TextSplitter:
    """
    Clase para dividir texto en chunks más pequeños.
    Mantiene la coherencia del texto dividiendo en párrafos cuando es posible.
    """
    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        """
        Inicializa el TextSplitter.
        
        Args:
            chunk_size (int): Tamaño máximo de cada chunk en caracteres
            overlap (int): Número de caracteres que se solapan entre chunks consecutivos
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def split(self, text: str) -> List[str]:
        """
        Divide el texto en chunks.
        
        Args:
            text (str): Texto a dividir
            
        Returns:
            List[str]: Lista de chunks de texto
        """
        # Primero dividimos por párrafos
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            # Si el párrafo es más largo que el chunk_size, lo dividimos
            if len(paragraph) > self.chunk_size:
                # Si ya tenemos contenido en el chunk actual, lo guardamos
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                
                # Dividimos el párrafo largo
                words = paragraph.split()
                temp_chunk = ""
                for word in words:
                    if len(temp_chunk) + len(word) + 1 <= self.chunk_size:
                        temp_chunk += word + " "
                    else:
                        chunks.append(temp_chunk.strip())
                        # Mantenemos un overlap de palabras
                        overlap_words = temp_chunk.split()[-self.overlap:]
                        temp_chunk = " ".join(overlap_words) + " " + word + " "
                
                if temp_chunk:
                    chunks.append(temp_chunk.strip())
            
            # Si el párrafo cabe en el chunk actual, lo añadimos
            elif len(current_chunk) + len(paragraph) + 2 <= self.chunk_size:
                current_chunk += paragraph + "\n\n"
            
            # Si no cabe, guardamos el chunk actual y empezamos uno nuevo
            else:
                chunks.append(current_chunk.strip())
                current_chunk = paragraph + "\n\n"
        
        # Añadimos el último chunk si hay contenido
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks

# Configuración de la base de datos
class Base(DeclarativeBase):
    pass

class Vector(Base):
    __tablename__ = 'vectors'

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    text: Mapped[str] = mapped_column(Text)
    vector = mapped_column(Vector(4096))  # Mistral usa embeddings de 4096 dimensiones
    metadata_: Mapped[dict | None] = mapped_column('metadata', JSONB)

    def __repr__(self):
        return f'Vector(id={self.id}, text={self.text[:50]}..., metadata={self.metadata_})'

DB_URL = 'postgresql+asyncpg://opc:P4ssw0rd!@localhost:5432/rag_db'
engine = create_async_engine(DB_URL)
Session = async_sessionmaker(engine, expire_on_commit=False)

async def db_create():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

# Cliente para Ollama Mistral
class OllamaClient:
    def __init__(self, base_url="http://localhost:11434"):
        self.base_url = base_url
        self.embedding_model = "mistral"
        self.llm_model = "mistral"
    
    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Obtiene embeddings de Mistral para una lista de textos"""
        embeddings = []
        async with aiohttp.ClientSession() as session:
            for text in texts:
                payload = {
                    "model": self.embedding_model,
                    "prompt": text
                }
                async with session.post(f"{self.base_url}/api/embeddings", json=payload) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        embeddings.append(data["embedding"])
                    else:
                        raise Exception(f"Error getting embedding: {await resp.text()}")
        return embeddings
    
    async def generate_response(self, messages: List[Dict[str, str]]) -> str:
        """Genera una respuesta usando Mistral"""
        payload = {
            "model": self.llm_model,
            "messages": messages,
            "stream": False
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.base_url}/api/chat", json=payload) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data["message"]["content"]
                else:
                    raise Exception(f"Error generating response: {await resp.text()}")

# Inicializar cliente Ollama
ollama = OllamaClient()

# Procesamiento de documentos
async def add_document_to_vector_db(doc_path: str):
    text = extract_text(doc_path)
    doc_name = os.path.splitext(os.path.basename(doc_path))[0]
    chunks = []
    text_splitter = TextSplitter(chunk_size=512)
    text_chunks = text_splitter.split(text)
    
    for idx, text_chunk in enumerate(text_chunks):
        chunks.append({
            'text': text_chunk,
            'metadata_': {'doc': doc_name, 'index': idx}
        })


    vectors = await ollama.get_embeddings([chunk['text'] for chunk in chunks])

    for chunk, vector in zip(chunks, vectors):
        chunk['vector'] = vector
    
    async with Session() as db:
        for chunk in chunks:
            db.add(Vector(**chunk))
        await db.commit()


async def vector_search(query_vector: List[float], top_k: int = 3) -> List[Dict[str, Any]]:
    async with Session() as db:
        query = (
            select(Vector.text, Vector.metadata_, Vector.vector.cosine_distance(query_vector).label('distance'))
            .order_by('distance')
            .limit(top_k)
        )
        res = await db.execute(query)
        return [{
            'text': text,
            'metadata': metadata,
            'score': 1 - distance
        } for text, metadata, distance in res]

# Sistema RAG con Mistral
SYSTEM_PROMPT = """
Eres un asistente de IA que responde preguntas sobre documentos en tu base de conocimiento.
Responde en el mismo idioma que la pregunta del usuario.
"""

RAG_PROMPT = """
Utiliza los siguientes fragmentos de contexto para responder la pregunta del usuario.
Debes usar solamente los hechos del contexto para responder.
Si la respuesta no puede encontrarse en el contexto, di que no tienes suficiente información para responder la pregunta y proporciona cualquier hecho relevante encontrado en el contexto.

Contexto:
{context}

Pregunta del usuario:
{question}
"""

async def answer_question_with_rag(question: str) -> str:
    # Obtener embedding de la pregunta
    query_vector = (await ollama.get_embeddings([question]))[0]
    
    # Buscar chunks relevantes
    top_chunks = await vector_search(query_vector, top_k=3)
    context = '\n\n---\n\n'.join([chunk['text'] for chunk in top_chunks]) + '\n\n---'
    
    # Preparar mensajes para Mistral
    user_message = RAG_PROMPT.format(context=context, question=question)
    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': user_message}
    ]
    
    # Generar respuesta
    return await ollama.generate_response(messages)

# Ejemplo de uso
async def main():
    
    await db_create()
    await add_document_to_vector_db("LISTAT.pdf")
 
    while True:
        try:
            question = input("\nTu pregunta: ").strip()
            
            # Comando para salir
            if question.lower() in ['salir', 'exit', 'quit']:
                print("\n👋 Sesión terminada\n")
                break
                
            # Validación pregunta vacía
            if not question:
                print("⚠️ Por favor escribe una pregunta válida")
                continue
                
            # Procesamiento
            print("\n🔄 Procesando tu pregunta...")
            answer = await answer_question_with_rag(question)
            
            # Formateo de respuesta
            print("\n" + "="*50)
            print(f"💡 Respuesta: {answer}")
            print("="*50)
            
        except KeyboardInterrupt:
            print("\n\n⚠️ Operación cancelada por el usuario")
            break
        except Exception as e:
            print(f"\n❌ Error procesando la pregunta: {str(e)}")

import asyncio
asyncio.run(main())





