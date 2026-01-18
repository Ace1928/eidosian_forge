from __future__ import annotations
import asyncio
import enum
import json
import logging
import struct
import uuid
from collections import OrderedDict
from enum import Enum
from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseSettings
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores.utils import maximal_marginal_relevance
def max_marginal_relevance_search_with_score_by_vector(self, embedding: List[float], k: int=4, fetch_k: int=20, lambda_mult: float=0.5, filter: Optional[Dict[str, str]]=None, **kwargs: Any) -> List[Tuple[Document, float]]:
    """Return docs selected using the maximal marginal relevance with score
            to embedding vector.

        Maximal marginal relevance optimizes for similarity to query AND diversity
            among selected documents.

        Args:
            embedding: Embedding to look up documents similar to.
            k (int): Number of Documents to return. Defaults to 4.
            fetch_k (int): Number of Documents to fetch to pass to MMR algorithm.
                Defaults to 20.
            lambda_mult (float): Number between 0 and 1 that determines the degree
                of diversity among the results with 0 corresponding
                to maximum diversity and 1 to minimum diversity.
                Defaults to 0.5.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.

        Returns:
            List[Tuple[Document, float]]: List of Documents selected by maximal marginal
                relevance to the query and score for each.
        """
    resp = self.__query_collection(embedding=embedding, k=fetch_k, filter=filter)
    records: OrderedDict = resp['records']
    results = list(zip(*list(records.values())))
    embedding_list = [struct.unpack('%sf' % self.dimensions, embedding) for embedding in records['embedding']]
    mmr_selected = maximal_marginal_relevance(np.array(embedding, dtype=np.float32), embedding_list, k=k, lambda_mult=lambda_mult)
    candidates = self._results_to_docs_and_scores(results)
    return [r for i, r in enumerate(candidates) if i in mmr_selected]