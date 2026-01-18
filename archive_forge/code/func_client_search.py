from __future__ import annotations
import uuid
import warnings
from typing import (
from langchain_core._api import deprecated
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.utils import get_from_dict_or_env
from langchain_core.vectorstores import VectorStore
def client_search(self, client: Any, index_name: str, script_query: Dict, size: int) -> Any:
    version_num = client.info()['version']['number'][0]
    version_num = int(version_num)
    if version_num >= 8:
        response = client.search(index=index_name, query=script_query, size=size)
    else:
        response = client.search(index=index_name, body={'query': script_query, 'size': size})
    return response