from __future__ import annotations
import enum
import logging
import uuid
from datetime import timedelta
from typing import (
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.utils import get_from_dict_or_env
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores.utils import DistanceStrategy
def delete_by_metadata(self, filter: Union[Dict[str, str], List[Dict[str, str]]], **kwargs: Any) -> Optional[bool]:
    """Delete by vector ID or other criteria.

        Args:
            ids: List of ids to delete.
            **kwargs: Other keyword arguments that subclasses might use.

        Returns:
            Optional[bool]: True if deletion is successful,
            False otherwise, None if not implemented.
        """
    self.sync_client.delete_by_metadata(filter)
    return True