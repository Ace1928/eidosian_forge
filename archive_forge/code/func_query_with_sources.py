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
def query_with_sources(self, question: str, llm: Optional[BaseLanguageModel]=None, retriever_kwargs: Optional[Dict[str, Any]]=None, **kwargs: Any) -> dict:
    """Query the vectorstore and get back sources."""
    llm = llm or OpenAI(temperature=0)
    retriever_kwargs = retriever_kwargs or {}
    chain = RetrievalQAWithSourcesChain.from_chain_type(llm, retriever=self.vectorstore.as_retriever(**retriever_kwargs), **kwargs)
    return chain.invoke({chain.question_key: question})