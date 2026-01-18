from __future__ import annotations
import json
import uuid
from typing import (
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
def marqo_bulk_similarity_search(self, queries: Iterable[Union[str, Dict[str, float]]], k: int=4) -> Dict[str, List[Dict[str, List[Dict[str, str]]]]]:
    """Return documents from Marqo using a bulk search, exposes Marqo's
        output directly

        Args:
            queries (Iterable[Union[str, Dict[str, float]]]): A list of queries.
            k (int, optional): The number of documents to return for each query.
            Defaults to 4.

        Returns:
            Dict[str, Dict[List[Dict[str, Dict[str, Any]]]]]: A bulk search results
            object
        """
    bulk_results = {'result': [self._client.index(self._index_name).search(q=query, searchable_attributes=self._searchable_attributes, limit=k) for query in queries]}
    return bulk_results