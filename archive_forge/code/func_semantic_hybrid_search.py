from __future__ import annotations
import base64
import json
import logging
import uuid
from typing import (
import numpy as np
from langchain_core.callbacks import (
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import root_validator
from langchain_core.retrievers import BaseRetriever
from langchain_core.utils import get_from_env
from langchain_core.vectorstores import VectorStore
def semantic_hybrid_search(self, query: str, k: int=4, **kwargs: Any) -> List[Document]:
    """
        Returns the most similar indexed documents to the query text.

        Args:
            query (str): The query text for which to find similar documents.
            k (int): The number of documents to return. Default is 4.

        Returns:
            List[Document]: A list of documents that are most similar to the query text.
        """
    docs_and_scores = self.semantic_hybrid_search_with_score_and_rerank(query, k=k, filters=kwargs.get('filters', None))
    return [doc for doc, _, _ in docs_and_scores]