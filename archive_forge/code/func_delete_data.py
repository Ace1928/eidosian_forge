from __future__ import annotations
from pydantic import BaseModel
from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Any, Dict, Optional, Iterable, Type, TYPE_CHECKING
def delete_data(self) -> None:
    """
        Deletes the Data
        """
    self.pdict.delete(self.data_cache_key)