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
def iter_contained_typevars(v: Any) -> Iterator[TypeVarType]:
    """Recursively iterate through all subtypes and type args of `v` and yield any typevars that are found.

    This is inspired as an alternative to directly accessing the `__parameters__` attribute of a GenericAlias,
    since __parameters__ of (nested) generic BaseModel subclasses won't show up in that list.
    """
    if isinstance(v, TypeVar):
        yield v
    elif is_model_class(v):
        yield from v.__pydantic_generic_metadata__['parameters']
    elif isinstance(v, (DictValues, list)):
        for var in v:
            yield from iter_contained_typevars(var)
    else:
        args = get_args(v)
        for arg in args:
            yield from iter_contained_typevars(arg)