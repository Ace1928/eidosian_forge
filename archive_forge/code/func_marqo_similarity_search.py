from __future__ import annotations
import json
import uuid
from typing import (
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
def marqo_similarity_search(self, query: Union[str, Dict[str, float]], k: int=4) -> Dict[str, List[Dict[str, str]]]:
    """Return documents from Marqo exposing Marqo's output directly

        Args:
            query (str): The query to search with.
            k (int, optional): The number of documents to return. Defaults to 4.

        Returns:
            List[Dict[str, Any]]: This hits from marqo.
        """
    results = self._client.index(self._index_name).search(q=query, searchable_attributes=self._searchable_attributes, limit=k)
    return results