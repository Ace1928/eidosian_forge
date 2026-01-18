from __future__ import annotations
import logging
from enum import Enum
from typing import (
import numpy as np
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores.utils import maximal_marginal_relevance
def delete_document_by_id(self, document_id: Optional[str]=None) -> None:
    """Removes a Specific Document by Id

        Args:
            document_id: The document identifier
        """
    try:
        from bson.objectid import ObjectId
    except ImportError as e:
        raise ImportError('Unable to import bson, please install with `pip install bson`.') from e
    if document_id is None:
        raise ValueError('No document id provided to delete.')
    self._collection.delete_one({'_id': ObjectId(document_id)})