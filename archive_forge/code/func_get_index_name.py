from __future__ import annotations
import logging
from enum import Enum
from typing import (
import numpy as np
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores.utils import maximal_marginal_relevance
def get_index_name(self) -> str:
    """Returns the index name

        Returns:
            Returns the index name

        """
    return self._index_name