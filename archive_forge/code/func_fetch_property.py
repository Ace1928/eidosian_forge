from __future__ import annotations
import sys
import copy
import pathlib
import inspect
import functools
import importlib.util
from typing import Any, Dict, Callable, Union, Optional, Type, TypeVar, List, Tuple, cast, TYPE_CHECKING
from types import ModuleType
def fetch_property(obj: Union[Type['BaseModel'], Dict], key: str, default: Optional[Any]=None):
    """
    Fetches a property from a dict or object
    """
    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)