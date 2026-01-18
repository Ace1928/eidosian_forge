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
@deprecated('0.0.1', alternative='similarity_search(distance_threshold=0.1)')
def similarity_search_limit_score(self, query: str, k: int=4, score_threshold: float=0.2, **kwargs: Any) -> List[Document]:
    """
        Returns the most similar indexed documents to the query text within the
        score_threshold range.

        Deprecated: Use similarity_search with distance_threshold instead.

        Args:
            query (str): The query text for which to find similar documents.
            k (int): The number of documents to return. Default is 4.
            score_threshold (float): The minimum matching *distance* required
                for a document to be considered a match. Defaults to 0.2.

        Returns:
            List[Document]: A list of documents that are most similar to the query text
                including the match score for each document.

        Note:
            If there are no documents that satisfy the score_threshold value,
            an empty list is returned.

        """
    return self.similarity_search(query, k=k, distance_threshold=score_threshold, **kwargs)