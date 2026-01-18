from __future__ import annotations
import abc
import atexit
import sqlite3
import pathlib
import filelock
import contextlib
from pydantic import BaseModel, Field, model_validator, model_serializer
from lazyops.imports._aiosqlite import resolve_aiosqlite
from lazyops.utils.lazy import lazy_import
from lazyops.utils import logger, Timer
from typing import Optional, List, Dict, Any, Union, Type, Tuple, TypeVar, AsyncGenerator, overload, TYPE_CHECKING
def reindex_from_internal_index(self, include: Optional[List[str]]=None, exclude: Optional[List[str]]=None, **kwargs) -> int:
    """
        Saves all items from the internal index to the DB
        """
    include = include or []
    exclude = exclude or []
    count = 0
    for tablename, index in self.internal_index.items():
        if (not include or tablename in include) and (not exclude or tablename not in exclude):
            items = list(index.values())
            if not items:
                continue
            self.set_many(items, skip_index=True, **kwargs)
            count += len(items)
    return count