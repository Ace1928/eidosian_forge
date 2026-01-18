from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type
from langchain_core.callbacks import (
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import Extra
from langchain_core.vectorstores import VectorStore
from langchain.chains.router.base import RouterChain
@classmethod
def from_names_and_descriptions(cls, names_and_descriptions: Sequence[Tuple[str, Sequence[str]]], vectorstore_cls: Type[VectorStore], embeddings: Embeddings, **kwargs: Any) -> EmbeddingRouterChain:
    """Convenience constructor."""
    documents = []
    for name, descriptions in names_and_descriptions:
        for description in descriptions:
            documents.append(Document(page_content=description, metadata={'name': name}))
    vectorstore = vectorstore_cls.from_documents(documents, embeddings)
    return cls(vectorstore=vectorstore, **kwargs)