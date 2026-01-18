from __future__ import annotations
from pydantic import BaseModel
from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Any, Dict, Optional, Iterable, Type, TYPE_CHECKING
@classmethod
def create_hash_key(cls, data: Any) -> str:
    """
        Creates a hash for the data
        """
    from ..utils.helpers import get_hashed_key
    return get_hashed_key(data)