from __future__ import annotations
import enum
import logging
import os
from hashlib import md5
from typing import (
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.utils import get_from_dict_or_env
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores.utils import DistanceStrategy
def sort_by_index_name(lst: List[Dict[str, Any]], index_name: str) -> List[Dict[str, Any]]:
    """Sort first element to match the index_name if exists"""
    return sorted(lst, key=lambda x: x.get('name') != index_name)