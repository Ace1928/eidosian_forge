import collections.abc
import copy
import dataclasses
import inspect
import sys
import types
import warnings
from typing import (
from typing_extensions import Annotated, Self, get_args, get_origin, get_type_hints
from . import _fields, _unsafe_cache
from ._typing import TypeForm
def narrow_collection_types(typ: TypeOrCallable, default_instance: Any) -> TypeOrCallable:
    """TypeForm narrowing for containers. Infers types of container contents."""
    if hasattr(default_instance, '__len__') and len(default_instance) == 0:
        return typ
    if typ is list and isinstance(default_instance, list):
        typ = List.__getitem__(Union.__getitem__(tuple(map(type, default_instance))))
    elif typ is set and isinstance(default_instance, set):
        typ = Set.__getitem__(Union.__getitem__(tuple(map(type, default_instance))))
    elif typ is tuple and isinstance(default_instance, tuple):
        typ = Tuple.__getitem__(tuple(map(type, default_instance)))
    return typ