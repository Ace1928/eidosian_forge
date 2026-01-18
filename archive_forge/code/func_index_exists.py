from __future__ import annotations
import logging
from enum import Enum
from typing import (
import numpy as np
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores.utils import maximal_marginal_relevance
def index_exists(self) -> bool:
    """Verifies if the specified index name during instance
            construction exists on the collection

        Returns:
          Returns True on success and False if no such index exists
            on the collection
        """
    cursor = self._collection.list_indexes()
    index_name = self._index_name
    for res in cursor:
        current_index_name = res.pop('name')
        if current_index_name == index_name:
            return True
    return False