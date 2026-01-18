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
def search_one(self, query: Optional[str]=None, skip: Optional[int]=None, return_id_only: Optional[bool]=None, return_fields: Optional[List[str]]=None, tablename: Optional[str]=None, schema_name: Optional[str]=None, cast_to_object: Optional[bool]=None, **kwargs) -> Optional[Union['SQLResultT', str]]:
    """
        Executes the sql search for one item
        """
    tablename = tablename or self.default_tablename
    schema = self.get_schema(tablename=tablename, schema_name=schema_name)
    if self.enable_internal_index and (return_id_only is None and return_fields is None) and self.internal_index.get(tablename):
        return_id_only = True
        if cast_to_object is None:
            cast_to_object = True
    query, kwargs = self.format_query(query, **kwargs)
    result = schema.search_one(conn=self.conn, query=query, skip=skip, return_id_only=return_id_only, return_fields=return_fields, tablename=tablename, **kwargs)
    if self.enable_internal_index and result and cast_to_object:
        return self.populate_item_from_index(tablename, result)
    return result