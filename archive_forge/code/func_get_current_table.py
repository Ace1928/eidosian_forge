from __future__ import annotations
import logging
import uuid
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Set, Tuple, Type
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores.utils import maximal_marginal_relevance
def get_current_table(self, **kwargs: Any) -> str:
    """Get the current table."""
    return self.using_table_name