from __future__ import annotations
import operator
import pickle
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_community.docstore.base import AddableMixin, Docstore
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores.utils import DistanceStrategy
def dependable_scann_import() -> Any:
    """
    Import `scann` if available, otherwise raise error.
    """
    try:
        import scann
    except ImportError:
        raise ImportError('Could not import scann python package. Please install it with `pip install scann` ')
    return scann