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
def check_typed_dict(value: Any, origin_type: Any, args: tuple[Any, ...], memo: TypeCheckMemo) -> None:
    if not isinstance(value, dict):
        raise TypeCheckError('is not a dict')
    declared_keys = frozenset(origin_type.__annotations__)
    if hasattr(origin_type, '__required_keys__'):
        required_keys = set(origin_type.__required_keys__)
    else:
        required_keys = set(declared_keys) if origin_type.__total__ else set()
    existing_keys = set(value)
    extra_keys = existing_keys - declared_keys
    if extra_keys:
        keys_formatted = ', '.join((f'"{key}"' for key in sorted(extra_keys, key=repr)))
        raise TypeCheckError(f'has unexpected extra key(s): {keys_formatted}')
    type_hints: dict[str, type] = {}
    for key, annotation in origin_type.__annotations__.items():
        if isinstance(annotation, ForwardRef):
            annotation = evaluate_forwardref(annotation, memo)
            if get_origin(annotation) is NotRequired:
                required_keys.discard(key)
                annotation = get_args(annotation)[0]
        type_hints[key] = annotation
    missing_keys = required_keys - existing_keys
    if missing_keys:
        keys_formatted = ', '.join((f'"{key}"' for key in sorted(missing_keys, key=repr)))
        raise TypeCheckError(f'is missing required key(s): {keys_formatted}')
    for key, argtype in type_hints.items():
        argvalue = value.get(key, _missing)
        if argvalue is not _missing:
            try:
                check_type_internal(argvalue, argtype, memo)
            except TypeCheckError as exc:
                exc.append_path_element(f'value of key {key!r}')
                raise