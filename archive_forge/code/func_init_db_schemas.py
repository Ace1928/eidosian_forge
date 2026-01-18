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
def init_db_schemas(self) -> None:
    """
        Initializes the database schemas
        """
    if self._schemas_initialized:
        return
    self.conn_prehook()
    conn = sqlite3.connect(self.sql_data_path.as_posix(), check_same_thread=False)
    self.init_schemas(conn)
    self.conn_posthook()