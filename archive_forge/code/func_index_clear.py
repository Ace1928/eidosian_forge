from __future__ import annotations
import json
import logging
import uuid
from typing import Any, Iterable, List, Optional, Tuple, Type, cast
import requests
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
def index_clear(self, cache_name: str) -> requests.Response:
    """Clear an index on a cache
        Args:
            cache_name(str): name of the cache.
        Returns:
            An http Response containing the result of the operation
        """
    api_url = self._default_node + self._cache_url + '/' + cache_name + '/search/indexes?action=clear'
    return requests.post(api_url, timeout=REST_TIMEOUT)