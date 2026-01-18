from __future__ import annotations
import json
import logging
import warnings
from typing import (
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
def get_dimensionality(self) -> int:
    """
        Function that does a dummy embedding to figure out how many dimensions
        this embedding function returns. Needed for the virtual table DDL.
        """
    dummy_text = 'This is a dummy text'
    dummy_embedding = self._embedding.embed_query(dummy_text)
    return len(dummy_embedding)