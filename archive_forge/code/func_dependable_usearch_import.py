from __future__ import annotations
from typing import Any, Dict, Iterable, List, Optional, Tuple
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_community.docstore.base import AddableMixin, Docstore
from langchain_community.docstore.in_memory import InMemoryDocstore
def dependable_usearch_import() -> Any:
    """
    Import usearch if available, otherwise raise error.
    """
    try:
        import usearch.index
    except ImportError:
        raise ImportError('Could not import usearch python package. Please install it with `pip install usearch` ')
    return usearch.index