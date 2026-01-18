from __future__ import annotations
import sys
import copy
import pathlib
import inspect
import functools
import importlib.util
from typing import Any, Dict, Callable, Union, Optional, Type, TypeVar, List, Tuple, cast, TYPE_CHECKING
from types import ModuleType
def get_obj_class_name(obj: Any, is_parent: Optional[bool]=None) -> str:
    """
    Returns the module name + class name of an object

    args:
        obj: the object to get the class name of
        is_parent: if True, then it treats the object as unitialized and gets the class name of the parent
    """
    if is_parent is None:
        is_parent = inspect.isclass(obj)
    if is_parent:
        return f'{obj.__module__}.{obj.__name__}'
    return f'{obj.__class__.__module__}.{obj.__class__.__name__}'