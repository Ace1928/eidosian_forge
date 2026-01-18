from __future__ import annotations
import uuid
import warnings
from typing import (
from langchain_core._api import deprecated
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.utils import get_from_dict_or_env
from langchain_core.vectorstores import VectorStore
def knn_search(self, query: Optional[str]=None, k: Optional[int]=10, query_vector: Optional[List[float]]=None, model_id: Optional[str]=None, size: Optional[int]=10, source: Optional[bool]=True, fields: Optional[Union[List[Mapping[str, Any]], Tuple[Mapping[str, Any], ...], None]]=None, page_content: Optional[str]='text') -> List[Tuple[Document, float]]:
    """
        Perform a k-NN search on the Elasticsearch index.

        Args:
            query (str, optional): The query text to search for.
            k (int, optional): The number of nearest neighbors to return.
            query_vector (List[float], optional): The query vector to search for.
            model_id (str, optional): The ID of the model to use for transforming the
                query text into a vector.
            size (int, optional): The number of search results to return.
            source (bool, optional): Whether to return the source of the search results.
            fields (List[Mapping[str, Any]], optional): The fields to return in the
                search results.
            page_content (str, optional): The name of the field that contains the page
                content.

        Returns:
            A list of tuples, where each tuple contains a Document object and a score.
        """
    if not source and (fields is None or not any((page_content in field for field in fields))):
        raise ValueError('If source=False `page_content` field must be in `fields`')
    knn_query_body = self._default_knn_query(query_vector=query_vector, query=query, model_id=model_id, k=k)
    response = self.client.search(index=self.index_name, knn=knn_query_body, size=size, source=source, fields=fields)
    hits = [hit for hit in response['hits']['hits']]
    docs_and_scores = [(Document(page_content=hit['_source'][page_content] if source else hit['fields'][page_content][0], metadata=hit['fields'] if fields else {}), hit['_score']) for hit in hits]
    return docs_and_scores