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
def sql_set(self, conn: 'sqlite3.Connection', tablename: Optional[str]=None, **kwargs):
    """
        Saves the data to the database
        """
    insert_dict, insert_data = self._get_export_sql_data_and_index(self.sql_schema, **kwargs)
    cur = conn.cursor()
    insert_script = SqliteTemplates['insert'].render(**self.sql_schema, **insert_dict)
    try:
        cur.execute(insert_script, insert_data)
        conn.commit()
        return cur.lastrowid
    except Exception as e:
        logger.error(f'[{self.sql_tablename}] Error in sql save: {insert_script}: {insert_data}: {e}')
        raise e