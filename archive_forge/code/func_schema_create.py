from __future__ import annotations
import json
import logging
import uuid
from typing import Any, Iterable, List, Optional, Tuple, Type, cast
import requests
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
def schema_create(self, proto: str) -> requests.Response:
    """Deploy the schema for the vector db
        Args:
            proto(str): protobuf schema
        Returns:
            An http Response containing the result of the operation
        """
    return self.ispn.schema_post(self._entity_name + '.proto', proto)