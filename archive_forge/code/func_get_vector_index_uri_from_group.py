from __future__ import annotations
import pickle
import random
import sys
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores.utils import maximal_marginal_relevance
def get_vector_index_uri_from_group(group: Any) -> str:
    """Get the URI of the vector index."""
    return group[VECTOR_INDEX_NAME].uri