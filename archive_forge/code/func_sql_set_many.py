from __future__ import annotations
import abc
import datetime
import contextlib
from pydantic import BaseModel, model_validator, Field, PrivateAttr, validator
from lazyops.utils.logs import logger
from .static import SqliteTemplates
from .registry import get_or_register_sqlite_schema, get_or_register_sqlite_connection, retrieve_sqlite_model_schema, get_sqlite_model_pkey, get_or_register_sqlite_tablename, SQLiteModelConfig, get_sqlite_model_config
from .utils import normalize_sql_text
from typing import Optional, List, Tuple, Dict, Union, TypeVar, Any, overload, TYPE_CHECKING
@classmethod
def sql_set_many(cls: type['SQLResultT'], conn: 'sqlite3.Connection', items: List['SQLResultT'], tablename: Optional[str]=None, include: 'IncEx'=None, exclude: 'IncEx'=None, by_alias: bool=False, exclude_unset: bool=False, exclude_defaults: bool=False, exclude_none: bool=False, **kwargs):
    """
        Inserts many items to the database
        """
    schemas = get_or_register_sqlite_schema(cls, tablename)
    sql_insert_fields = items[0]._get_export_sql_insert_fields(schemas, include=include, exclude=exclude, by_alias=by_alias, exclude_unset=exclude_unset, exclude_defaults=exclude_defaults, exclude_none=exclude_none)
    sql_insert_values = [item._get_export_sql_data(schemas, include=include, exclude=exclude, by_alias=by_alias, exclude_unset=exclude_unset, exclude_defaults=exclude_defaults, exclude_none=exclude_none) for item in items]
    insert_script = SqliteTemplates['insert'].render(**schemas, sql_insert_fields=sql_insert_fields)
    cur = conn.cursor()
    cur.executemany(insert_script, sql_insert_values)
    conn.commit()
    return cur.lastrowid