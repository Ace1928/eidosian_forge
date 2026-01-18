from typing import Any, Dict, List, Optional, Type
from langchain_community.document_loaders.base import BaseLoader
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.llms.openai import OpenAI
from langchain_community.vectorstores.inmemory import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel
from langchain_core.pydantic_v1 import BaseModel, Extra, Field
from langchain_core.vectorstores import VectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter
from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
from langchain.chains.retrieval_qa.base import RetrievalQA
def from_documents(self, documents: List[Document]) -> VectorStoreIndexWrapper:
    """Create a vectorstore index from documents."""
    sub_docs = self.text_splitter.split_documents(documents)
    vectorstore = self.vectorstore_cls.from_documents(sub_docs, self.embedding, **self.vectorstore_kwargs)
    return VectorStoreIndexWrapper(vectorstore=vectorstore)