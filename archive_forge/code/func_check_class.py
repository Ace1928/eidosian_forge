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
def check_class(value: Any, origin_type: Any, args: tuple[Any, ...], memo: TypeCheckMemo) -> None:
    if not isclass(value) and (not isinstance(value, generic_alias_types)):
        raise TypeCheckError('is not a class')
    if not args:
        return
    if isinstance(args[0], ForwardRef):
        expected_class = evaluate_forwardref(args[0], memo)
    else:
        expected_class = args[0]
    if expected_class is Any:
        return
    elif getattr(expected_class, '_is_protocol', False):
        check_protocol(value, expected_class, (), memo)
    elif isinstance(expected_class, TypeVar):
        check_typevar(value, expected_class, (), memo, subclass_check=True)
    elif get_origin(expected_class) is Union:
        errors: dict[str, TypeCheckError] = {}
        for arg in get_args(expected_class):
            if arg is Any:
                return
            try:
                check_class(value, type, (arg,), memo)
                return
            except TypeCheckError as exc:
                errors[get_type_name(arg)] = exc
        else:
            formatted_errors = indent('\n'.join((f'{key}: {error}' for key, error in errors.items())), '  ')
            raise TypeCheckError(f'did not match any element in the union:\n{formatted_errors}')
    elif not issubclass(value, expected_class):
        raise TypeCheckError(f'is not a subclass of {qualified_name(expected_class)}')