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
def check_typevar(value: Any, origin_type: TypeVar, args: tuple[Any, ...], memo: TypeCheckMemo, *, subclass_check: bool=False) -> None:
    if origin_type.__bound__ is not None:
        annotation = Type[origin_type.__bound__] if subclass_check else origin_type.__bound__
        check_type_internal(value, annotation, memo)
    elif origin_type.__constraints__:
        for constraint in origin_type.__constraints__:
            annotation = Type[constraint] if subclass_check else constraint
            try:
                check_type_internal(value, annotation, memo)
            except TypeCheckError:
                pass
            else:
                break
        else:
            formatted_constraints = ', '.join((get_type_name(constraint) for constraint in origin_type.__constraints__))
            raise TypeCheckError(f'does not match any of the constraints ({formatted_constraints})')