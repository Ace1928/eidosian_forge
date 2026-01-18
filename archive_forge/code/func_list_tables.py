from __future__ import annotations
import logging
import uuid
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Set, Tuple, Type
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores.utils import maximal_marginal_relevance
def list_tables(self, **kwargs: Any) -> List[str]:
    """List all the tables created by the client."""
    if self.awadb_client is None:
        return []
    return self.awadb_client.ListAllTables()