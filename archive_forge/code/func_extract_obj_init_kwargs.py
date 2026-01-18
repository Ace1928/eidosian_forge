from __future__ import annotations
import sys
import copy
import pathlib
import inspect
import functools
import importlib.util
from typing import Any, Dict, Callable, Union, Optional, Type, TypeVar, List, Tuple, cast, TYPE_CHECKING
from types import ModuleType
def extract_obj_init_kwargs(obj: object) -> List[str]:
    """
    Extracts the kwargs that are valid for an object
    """
    global _valid_class_init_kwarg
    obj_name = get_obj_class_name(obj)
    if obj_name not in _valid_class_init_kwarg:
        argspec = inspect.getfullargspec(obj.__init__)
        _args = list(argspec.args)
        _args.extend(iter(argspec.kwonlyargs))
        if hasattr(obj, '__bases__'):
            for base in obj.__bases__:
                _args.extend(extract_obj_init_kwargs(base))
        _valid_class_init_kwarg[obj_name] = list(set(_args))
    return _valid_class_init_kwarg[obj_name]