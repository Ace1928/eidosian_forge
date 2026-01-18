from __future__ import annotations
import sys
import types
import typing
from collections import ChainMap
from contextlib import contextmanager
from contextvars import ContextVar
from types import prepare_class
from typing import TYPE_CHECKING, Any, Iterator, List, Mapping, MutableMapping, Tuple, TypeVar
from weakref import WeakValueDictionary
import typing_extensions
from ._core_utils import get_type_ref
from ._forward_ref import PydanticRecursiveRef
from ._typing_extra import TypeVarType, typing_base
from ._utils import all_identical, is_model_class
def has_instance_in_type(type_: Any, isinstance_target: Any) -> bool:
    """Checks if the type, or any of its arbitrary nested args, satisfy
    `isinstance(<type>, isinstance_target)`.
    """
    if isinstance(type_, isinstance_target):
        return True
    type_args = get_args(type_)
    origin_type = get_origin(type_)
    if origin_type is typing_extensions.Annotated:
        annotated_type, *annotations = type_args
        return has_instance_in_type(annotated_type, isinstance_target)
    if any((has_instance_in_type(a, isinstance_target) for a in type_args)):
        return True
    if isinstance(type_, (List, list)) and (not isinstance(type_, typing_extensions.ParamSpec)):
        if any((has_instance_in_type(element, isinstance_target) for element in type_)):
            return True
    return False