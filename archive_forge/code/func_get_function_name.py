from __future__ import annotations
import sys
import copy
import pathlib
import inspect
import functools
import importlib.util
from typing import Any, Dict, Callable, Union, Optional, Type, TypeVar, List, Tuple, cast, TYPE_CHECKING
from types import ModuleType
def get_function_name(func: Callable) -> str:
    """
    Returns the module name + function name of a function
    """
    func = inspect.unwrap(func)
    return f'{func.__module__}.{func.__name__}'