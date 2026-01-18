from __future__ import annotations
import json
import logging
import uuid
from typing import Any, Iterable, List, Optional, Tuple, Type, cast
import requests
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
def cache_delete(self, name: str) -> requests.Response:
    """Delete a cache
        Args:
            name(str): name of the cache.
        Returns:
            An http Response containing the result of the operation
        """
    api_url = self._default_node + self._cache_url + '/' + name
    response = requests.delete(api_url, timeout=REST_TIMEOUT)
    return response