from __future__ import annotations
import logging
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores.utils import maximal_marginal_relevance
@classmethod
def force_delete_by_path(cls, path: str) -> None:
    """Force delete dataset by path.

        Args:
            path (str): path of the dataset to delete.

        Raises:
            ValueError: if deeplake is not installed.
        """
    try:
        import deeplake
    except ImportError:
        raise ValueError('Could not import deeplake python package. Please install it with `pip install deeplake`.')
    deeplake.delete(path, large_ok=True, force=True)