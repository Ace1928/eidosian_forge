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
def similarity_search(self, query: str, k: int=4, filter: Optional[RedisFilterExpression]=None, return_metadata: bool=True, distance_threshold: Optional[float]=None, **kwargs: Any) -> List[Document]:
    """Run similarity search

        Args:
            query (str): The query text for which to find similar documents.
            k (int): The number of documents to return. Default is 4.
            filter (RedisFilterExpression, optional): Optional metadata filter.
                Defaults to None.
            return_metadata (bool, optional): Whether to return metadata.
                Defaults to True.
            distance_threshold (Optional[float], optional): Maximum vector distance
                between selected documents and the query vector. Defaults to None.

        Returns:
            List[Document]: A list of documents that are most similar to the query
                text.
        """
    query_embedding = self._embeddings.embed_query(query)
    return self.similarity_search_by_vector(query_embedding, k=k, filter=filter, return_metadata=return_metadata, distance_threshold=distance_threshold, **kwargs)