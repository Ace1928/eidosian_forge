from __future__ import annotations
import os
import pickle
import uuid
from configparser import ConfigParser
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_community.docstore.base import Docstore
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores.utils import maximal_marginal_relevance
def similarity_search_with_score_by_index(self, docstore_index: int, k: int=4, search_k: int=-1) -> List[Tuple[Document, float]]:
    """Return docs most similar to query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            search_k: inspect up to search_k nodes which defaults
                to n_trees * n if not provided
        Returns:
            List of Documents most similar to the query and score for each
        """
    idxs, dists = self.index.get_nns_by_item(docstore_index, k, search_k=search_k, include_distances=True)
    return self.process_index_results(idxs, dists)