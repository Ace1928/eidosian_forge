from __future__ import annotations
import base64
import logging
import uuid
from copy import deepcopy
from typing import (
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores.utils import maximal_marginal_relevance
def query_collection_embeddings(self, query_embeddings: Optional[List[List[float]]]=None, collection_name: Optional[str]=None, n_results: int=DEFAULT_K, fetch_k: int=DEFAULT_FETCH_K, filter: Union[None, Dict[str, Any]]=None, results: Union[None, Dict[str, Any]]=None, normalize_distance: bool=False, **kwargs: Any) -> List[Tuple[Dict[str, Any], List]]:
    all_responses: List[Any] = []
    if collection_name is None:
        collection_name = self._collection_name
    if query_embeddings is None:
        return all_responses
    include = kwargs.get('include', ['metadatas'])
    if results is None and 'metadatas' in include:
        results = {'list': self.collection_properties, 'blob': 'embeddings' in include}
    for qemb in query_embeddings:
        response, response_array = self.get_descriptor_response('FindDescriptor', collection_name, k_neighbors=n_results, fetch_k=fetch_k, constraints=filter, results=results, normalize_distance=normalize_distance, query_embedding=qemb)
        all_responses.append([response, response_array])
    return all_responses