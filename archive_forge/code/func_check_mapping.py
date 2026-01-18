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
def check_mapping(value: Any, origin_type: Any, args: tuple[Any, ...], memo: TypeCheckMemo) -> None:
    if origin_type is Dict or origin_type is dict:
        if not isinstance(value, dict):
            raise TypeCheckError('is not a dict')
    if origin_type is MutableMapping or origin_type is collections.abc.MutableMapping:
        if not isinstance(value, collections.abc.MutableMapping):
            raise TypeCheckError('is not a mutable mapping')
    elif not isinstance(value, collections.abc.Mapping):
        raise TypeCheckError('is not a mapping')
    if args:
        key_type, value_type = args
        if key_type is not Any or value_type is not Any:
            samples = memo.config.collection_check_strategy.iterate_samples(value.items())
            for k, v in samples:
                try:
                    check_type_internal(k, key_type, memo)
                except TypeCheckError as exc:
                    exc.append_path_element(f'key {k!r}')
                    raise
                try:
                    check_type_internal(v, value_type, memo)
                except TypeCheckError as exc:
                    exc.append_path_element(f'value of key {k!r}')
                    raise