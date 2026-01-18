from __future__ import annotations
import logging
import os
import uuid
from typing import (
import numpy as np
import yaml
from langchain_core._api import deprecated
from langchain_core.callbacks import (
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.utils import get_from_dict_or_env
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever
from langchain_community.utilities.redis import (
from langchain_community.vectorstores.redis.constants import (
from langchain_community.vectorstores.utils import maximal_marginal_relevance
def max_marginal_relevance_search(self, query: str, k: int=4, fetch_k: int=20, lambda_mult: float=0.5, filter: Optional[RedisFilterExpression]=None, return_metadata: bool=True, distance_threshold: Optional[float]=None, **kwargs: Any) -> List[Document]:
    """Return docs selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity
            among selected documents.

        Args:
            query (str): Text to look up documents similar to.
            k (int): Number of Documents to return. Defaults to 4.
            fetch_k (int): Number of Documents to fetch to pass to MMR algorithm.
            lambda_mult (float): Number between 0 and 1 that determines the degree
                of diversity among the results with 0 corresponding
                to maximum diversity and 1 to minimum diversity.
                Defaults to 0.5.
            filter (RedisFilterExpression, optional): Optional metadata filter.
                Defaults to None.
            return_metadata (bool, optional): Whether to return metadata.
                Defaults to True.
            distance_threshold (Optional[float], optional): Maximum vector distance
                between selected documents and the query vector. Defaults to None.

        Returns:
            List[Document]: A list of Documents selected by maximal marginal relevance.
        """
    query_embedding = self._embeddings.embed_query(query)
    prefetch_docs = self.similarity_search_by_vector(query_embedding, k=fetch_k, filter=filter, return_metadata=return_metadata, distance_threshold=distance_threshold, **kwargs)
    prefetch_ids = [doc.metadata['id'] for doc in prefetch_docs]
    prefetch_embeddings = [_buffer_to_array(cast(bytes, self.client.hget(prefetch_id, self._schema.content_vector_key)), dtype=self._schema.vector_dtype) for prefetch_id in prefetch_ids]
    selected_indices = maximal_marginal_relevance(np.array(query_embedding), prefetch_embeddings, lambda_mult=lambda_mult, k=k)
    selected_docs = [prefetch_docs[i] for i in selected_indices]
    return selected_docs