from __future__ import annotations
import uuid
from typing import (
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.utils import xor_args
from langchain_core.vectorstores import VectorStore
def delete_cluster(self) -> None:
    """Delete the cluster."""
    self._client.delete_cluster(self._cluster.name)