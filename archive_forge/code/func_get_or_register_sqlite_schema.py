import functools
from pydantic import BaseModel, computed_field
from typing import Any, Dict, List, Optional, Type, Union, TYPE_CHECKING
def get_or_register_sqlite_schema(model: 'SQLiteModelMixin', tablename: Optional[str]=None, auto_set: Optional[bool]=None, conn: Optional[Union['sqlite3.Connection', 'aiosqlite.Connection']]=None) -> Dict[str, Union[str, List[str], Dict[str, Union[str, int]]]]:
    """
    Registers the SQLite Schema
    """
    global _sqlite_model_schema_registry
    tablename = get_or_register_sqlite_tablename(model, tablename)
    if tablename not in _sqlite_model_schema_registry:
        sql_fields, sql_pkey, search_precisions = model._get_sql_field_schema()
        sql_keys = list(sql_fields.keys())
        sql_insert_q = ('?, ' * len(sql_keys)).rstrip(', ')
        sql_insert = ', '.join(sql_keys)
        _sqlite_model_schema_registry[tablename] = {'tablename': tablename, 'sql_fields': sql_fields, 'sql_pkey': sql_pkey, 'sql_keys': sql_keys, 'sql_insert': sql_insert, 'sql_insert_q': sql_insert_q, 'search_precisions': search_precisions}
        if conn is not None:
            get_or_register_sqlite_connection(model, conn)
    if auto_set is not None:
        _sqlite_model_schema_registry[tablename]['auto_set'] = auto_set
    return _sqlite_model_schema_registry[tablename]