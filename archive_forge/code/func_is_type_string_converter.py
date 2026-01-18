import collections.abc
import dataclasses
import enum
import inspect
import os
import pathlib
from collections import deque
from typing import (
from typing_extensions import Annotated, Final, Literal, get_args, get_origin
from . import _resolver
from . import _strings
from ._typing import TypeForm
from .conf import _markers
def is_type_string_converter(typ: Union[Callable, TypeForm[Any]]) -> bool:
    """Check if type is a string converter, i.e., (arg: Union[str, Any]) -> T."""
    param_count = 0
    has_var_positional = False
    try:
        signature = inspect.signature(typ)
    except ValueError:
        return True
    type_annotations = _resolver.get_type_hints_with_nicer_errors(typ)
    for i, param in enumerate(signature.parameters.values()):
        annotation = type_annotations.get(param.name, param.annotation)
        annotation = _resolver.apply_type_from_typevar(annotation, {})
        if i == 0 and (not (get_origin(annotation) is Union and str in get_args(annotation) or annotation in (str, inspect.Parameter.empty))):
            return False
        if param.kind is inspect.Parameter.VAR_POSITIONAL:
            has_var_positional = True
        elif param.default is inspect.Parameter.empty and param.kind is not inspect.Parameter.VAR_KEYWORD:
            param_count += 1
    if not (param_count == 1 or (param_count == 0 and has_var_positional)):
        return False
    return True