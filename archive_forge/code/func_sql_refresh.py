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
def sql_refresh(self: 'SQLResultT', conn: 'sqlite3.Connection', tablename: Optional[str]=None, **kwargs) -> 'SQLResultT':
    """
        Refreshes the data from the database
        """
    schemas = get_or_register_sqlite_schema(self.__class__, tablename)
    refresh_script = SqliteTemplates['refresh'].render(**schemas)
    cur = conn.cursor()
    cur.execute(refresh_script, (getattr(self, schemas['sql_pkey']),))
    result = cur.fetchone()
    if result is None:
        return None
    data = dict(zip(schemas['sql_keys'], result))
    with self._sql_update_context():
        for key, value in data.items():
            if value is None:
                continue
            if key in schemas['sql_keys']:
                setattr(self, key, value)
    return self