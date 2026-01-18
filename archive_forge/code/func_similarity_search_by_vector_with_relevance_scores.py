from __future__ import annotations
import uuid
import warnings
from itertools import repeat
from typing import (
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores.utils import maximal_marginal_relevance
def similarity_search_by_vector_with_relevance_scores(self, query: List[float], k: int, filter: Optional[Dict[str, Any]]=None, postgrest_filter: Optional[str]=None, score_threshold: Optional[float]=None) -> List[Tuple[Document, float]]:
    match_documents_params = self.match_args(query, filter)
    query_builder = self._client.rpc(self.query_name, match_documents_params)
    if postgrest_filter:
        query_builder.params = query_builder.params.set('and', f'({postgrest_filter})')
    query_builder.params = query_builder.params.set('limit', k)
    res = query_builder.execute()
    match_result = [(Document(metadata=search.get('metadata', {}), page_content=search.get('content', '')), search.get('similarity', 0.0)) for search in res.data if search.get('content')]
    if score_threshold is not None:
        match_result = [(doc, similarity) for doc, similarity in match_result if similarity >= score_threshold]
        if len(match_result) == 0:
            warnings.warn(f'No relevant docs were retrieved using the relevance score threshold {score_threshold}')
    return match_result