import functools
from pydantic import BaseModel, computed_field
from typing import Any, Dict, List, Optional, Type, Union, TYPE_CHECKING
@functools.lru_cache()
def retrieve_sqlite_model_schema(model_name: str) -> Dict[str, Union[str, List[str], Dict[str, Union[str, int]]]]:
    """
    Retrieves the SQLite Model Schema
    """
    if model_name not in _sqlite_model_name_to_tablename:
        raise ValueError(f'Model {model_name} not registered')
    tablename = _sqlite_model_name_to_tablename[model_name]
    if tablename not in _sqlite_model_schema_registry:
        raise ValueError(f'Model {model_name} not registered')
    return _sqlite_model_schema_registry[tablename]