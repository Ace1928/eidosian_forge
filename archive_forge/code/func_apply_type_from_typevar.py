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
def apply_type_from_typevar(typ: TypeOrCallable, type_from_typevar: Dict[TypeVar, TypeForm[Any]]) -> TypeOrCallable:
    if typ in type_from_typevar:
        return type_from_typevar[typ]
    origin = get_origin(typ)
    args = get_args(typ)
    if len(args) > 0:
        if origin is Annotated:
            args = args[:1]
        if origin is collections.abc.Callable:
            assert isinstance(args[0], list)
            args = tuple(args[0]) + args[1:]
        if sys.version_info[:2] >= (3, 9):
            shim_table = {tuple: Tuple, list: List, dict: Dict, set: Set, frozenset: FrozenSet}
            if hasattr(types, 'UnionType'):
                shim_table[types.UnionType] = Union
            for new, old in shim_table.items():
                if isinstance(typ, new) or origin is new:
                    typ = old.__getitem__(args)
        return typ.copy_with(tuple((apply_type_from_typevar(x, type_from_typevar) for x in args)))
    return typ