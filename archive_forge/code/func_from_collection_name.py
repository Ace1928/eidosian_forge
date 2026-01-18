from __future__ import annotations
import uuid
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple, Union
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
@classmethod
def from_collection_name(cls, embedding: Embeddings, db_url: str, collection_name: str) -> PGVecto_rs:
    """Create new empty vectorstore with collection_name.
        Or connect to an existing vectorstore in database if exists.
        Arguments should be the same as when the vectorstore was created."""
    sample_embedding = embedding.embed_query('Hello pgvecto_rs!')
    return cls(embedding=embedding, dimension=len(sample_embedding), db_url=db_url, collection_name=collection_name)