from __future__ import annotations
import collections.abc
import inspect
import sys
import types
import typing
import warnings
from enum import Enum
from inspect import Parameter, isclass, isfunction
from io import BufferedIOBase, IOBase, RawIOBase, TextIOBase
from textwrap import indent
from typing import (
from unittest.mock import Mock
from ._config import ForwardRefPolicy
from ._exceptions import TypeCheckError, TypeHintWarning
from ._memo import TypeCheckMemo
from ._utils import evaluate_forwardref, get_stacklevel, get_type_name, qualified_name
def builtin_checker_lookup(origin_type: Any, args: tuple[Any, ...], extras: tuple[Any, ...]) -> TypeCheckerCallable | None:
    checker = origin_type_checkers.get(origin_type)
    if checker is not None:
        return checker
    elif is_typeddict(origin_type):
        return check_typed_dict
    elif isclass(origin_type) and issubclass(origin_type, Tuple):
        return check_tuple
    elif getattr(origin_type, '_is_protocol', False):
        return check_protocol
    elif isinstance(origin_type, ParamSpec):
        return check_paramspec
    elif isinstance(origin_type, TypeVar):
        return check_typevar
    elif origin_type.__class__ is NewType:
        return check_newtype
    elif isfunction(origin_type) and getattr(origin_type, '__module__', None) == 'typing' and getattr(origin_type, '__qualname__', '').startswith('NewType.') and hasattr(origin_type, '__supertype__'):
        return check_newtype
    return None