import functools
from pydantic import BaseModel, computed_field
from typing import Any, Dict, List, Optional, Type, Union, TYPE_CHECKING
def has_auto_set_in_schema(model: 'SQLiteModelMixin') -> bool:
    """
    Checks if the model has auto set in the schema
    """
    return 'auto_set' in get_or_register_sqlite_schema(model)