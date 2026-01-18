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
def check_callable(value: Any, origin_type: Any, args: tuple[Any, ...], memo: TypeCheckMemo) -> None:
    if not callable(value):
        raise TypeCheckError('is not callable')
    if args:
        try:
            signature = inspect.signature(value)
        except (TypeError, ValueError):
            return
        argument_types = args[0]
        if isinstance(argument_types, list) and (not any((type(item) is ParamSpec for item in argument_types))):
            unfulfilled_kwonlyargs = [param.name for param in signature.parameters.values() if param.kind == Parameter.KEYWORD_ONLY and param.default == Parameter.empty]
            if unfulfilled_kwonlyargs:
                raise TypeCheckError(f'has mandatory keyword-only arguments in its declaration: {', '.join(unfulfilled_kwonlyargs)}')
            num_positional_args = num_mandatory_pos_args = 0
            has_varargs = False
            for param in signature.parameters.values():
                if param.kind in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD):
                    num_positional_args += 1
                    if param.default is Parameter.empty:
                        num_mandatory_pos_args += 1
                elif param.kind == Parameter.VAR_POSITIONAL:
                    has_varargs = True
            if num_mandatory_pos_args > len(argument_types):
                raise TypeCheckError(f'has too many mandatory positional arguments in its declaration; expected {len(argument_types)} but {num_mandatory_pos_args} mandatory positional argument(s) declared')
            elif not has_varargs and num_positional_args < len(argument_types):
                raise TypeCheckError(f'has too few arguments in its declaration; expected {len(argument_types)} but {num_positional_args} argument(s) declared')