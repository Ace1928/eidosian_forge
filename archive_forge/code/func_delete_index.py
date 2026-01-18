from __future__ import annotations
import logging
from enum import Enum
from typing import (
import numpy as np
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores.utils import maximal_marginal_relevance
def delete_index(self) -> None:
    """Deletes the index specified during instance construction if it exists"""
    if self.index_exists():
        self._collection.drop_index(self._index_name)