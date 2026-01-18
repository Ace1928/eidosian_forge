from __future__ import annotations
import json
import logging
from hashlib import sha1
from threading import Thread
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseSettings
from langchain_core.vectorstores import VectorStore
def has_mul_sub_str(s: str, *args: Any) -> bool:
    """
    Check if a string contains multiple substrings.
    Args:
        s: string to check.
        *args: substrings to check.

    Returns:
        True if all substrings are in the string, False otherwise.
    """
    for a in args:
        if a not in s:
            return False
    return True