from __future__ import annotations
import uuid
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import (
import numpy as np
from langchain_core._api.deprecation import deprecated
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.runnables.utils import gather_with_concurrency
from langchain_core.utils.iter import batch_iterate
from langchain_core.vectorstores import VectorStore
from langchain_community.utilities.astradb import (
from langchain_community.vectorstores.utils import maximal_marginal_relevance
def delete_by_document_id(self, document_id: str) -> bool:
    """
        Remove a single document from the store, given its document ID.

        Args:
            document_id: The document ID

        Returns
            True if a document has indeed been deleted, False if ID not found.
        """
    self.astra_env.ensure_db_setup()
    deletion_response = self.collection.delete_one(document_id)
    return ((deletion_response or {}).get('status') or {}).get('deletedCount', 0) == 1