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
def dependable_tiledb_import() -> Any:
    """Import tiledb-vector-search if available, otherwise raise error."""
    try:
        import tiledb as tiledb
        import tiledb.vector_search as tiledb_vs
    except ImportError:
        raise ValueError('Could not import tiledb-vector-search python package. Please install it with `conda install -c tiledb tiledb-vector-search` or `pip install tiledb-vector-search`')
    return (tiledb_vs, tiledb)