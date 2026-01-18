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
def get_literal_args(literal_args: tuple[Any, ...]) -> tuple[Any, ...]:
    retval: list[Any] = []
    for arg in literal_args:
        if _is_literal_type(get_origin(arg)):
            retval.extend(get_literal_args(arg.__args__))
        elif arg is None or isinstance(arg, (int, str, bytes, bool, Enum)):
            retval.append(arg)
        else:
            raise TypeError(f'Illegal literal value: {arg}')
    return tuple(retval)