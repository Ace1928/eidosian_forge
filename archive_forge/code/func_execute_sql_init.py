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
def execute_sql_init(cls, conn: 'sqlite3.Connection', tablename: Optional[str]=None, skip_index: Optional[bool]=None, auto_set: Optional[bool]=None):
    """
        Executes the sql init

        Parameters
        ----------
        conn : sqlite3.Connection
            The connection to the database
        tablename : Optional[str], optional
            The table name to use, by default None. This must be passed at least once during initialization.
        skip_index : Optional[bool], optional
            Whether to skip the index creation, by default None. This must be passed at least once during initialization.
        auto_set : Optional[bool], optional
            Whether to automatically update the table whenever model attributes are updated, by default None.
        """
    cur = conn.cursor()
    schemas = get_or_register_sqlite_schema(cls, tablename, auto_set, conn)
    tablename = tablename or schemas['tablename']
    script = SqliteTemplates['init'].render(**schemas)
    try:
        cur.executescript(script)
        conn.commit()
    except Exception as e:
        logger.error(f'[{tablename}] Error in sql init: {script}: {e}')
        raise e
    if not skip_index and (index_items := cls._get_sql_index_items()):
        index_script = SqliteTemplates['index'].render(**schemas)
        try:
            cur.executemany(index_script, index_items)
            conn.commit()
        except Exception as e:
            logger.error(f'[{tablename}] Error in indexing: {index_script}: {index_items[0]}: {e}')
            raise e