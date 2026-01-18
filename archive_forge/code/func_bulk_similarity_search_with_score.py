from __future__ import annotations
import json
import uuid
from typing import (
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
def bulk_similarity_search_with_score(self, queries: Iterable[Union[str, Dict[str, float]]], k: int=4, **kwargs: Any) -> List[List[Tuple[Document, float]]]:
    """Return documents from Marqo that are similar to the query as well as
        their scores using a batch of queries.

        Args:
            query (Iterable[Union[str, Dict[str, float]]]): An iterable of queries
            to execute in bulk, queries in the list can be strings or dictionaries
            of weighted queries.
            k (int, optional): The number of documents to return. Defaults to 4.

        Returns:
            List[Tuple[Document, float]]: A list of lists of the matching
            documents and their scores for each query
        """
    bulk_results = self.marqo_bulk_similarity_search(queries=queries, k=k)
    bulk_documents: List[List[Tuple[Document, float]]] = []
    for results in bulk_results['result']:
        documents = self._construct_documents_from_results_with_score(results)
        bulk_documents.append(documents)
    return bulk_documents