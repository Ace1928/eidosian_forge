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
def check_if_not_null(props: List[str], values: List[Any]) -> None:
    """Check if the values are not None or empty string"""
    for prop, value in zip(props, values):
        if not value:
            raise ValueError(f'Parameter `{prop}` must not be None or empty string')