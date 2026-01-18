from __future__ import annotations
from .. import mesonlib, mlog
from .disabler import Disabler
from .exceptions import InterpreterException, InvalidArguments
from ._unholder import _unholder
from dataclasses import dataclass
from functools import wraps
import abc
import itertools
import copy
import typing as T
def check_value_type(types_tuple: T.Tuple[T.Union[T.Type, ContainerTypeInfo], ...], value: T.Any) -> bool:
    for t in types_tuple:
        if isinstance(t, ContainerTypeInfo):
            if t.check(value):
                return True
        elif isinstance(value, t):
            return True
    return False