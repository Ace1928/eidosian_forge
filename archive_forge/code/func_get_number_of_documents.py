from __future__ import annotations
import json
import uuid
from typing import (
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
def get_number_of_documents(self) -> int:
    """Helper to see the number of documents in the index

        Returns:
            int: The number of documents
        """
    return self._client.index(self._index_name).get_stats()['numberOfDocuments']