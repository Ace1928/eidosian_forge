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
def similarity_search_by_index(self, docstore_index: int, k: int=4, search_k: int=-1, **kwargs: Any) -> List[Document]:
    """Return docs most similar to docstore_index.

        Args:
            docstore_index: Index of document in docstore
            k: Number of Documents to return. Defaults to 4.
            search_k: inspect up to search_k nodes which defaults
                to n_trees * n if not provided

        Returns:
            List of Documents most similar to the embedding.
        """
    docs_and_scores = self.similarity_search_with_score_by_index(docstore_index, k, search_k)
    return [doc for doc, _ in docs_and_scores]