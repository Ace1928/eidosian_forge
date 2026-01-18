from __future__ import annotations
import dataclasses
import enum
import functools
from typing import IO, TYPE_CHECKING, Any, Optional, Set, Type, TypeVar, Union
from typing_extensions import get_args, get_origin
from .. import _fields, _resolver
def handle_type(typ: Type[Any]) -> Set[Type[Any]]:
    if _resolver.is_dataclass(typ) and typ not in parent_contained_dataclasses:
        return _get_contained_special_types_from_type(typ, _parent_contained_dataclasses=contained_special_types | parent_contained_dataclasses)
    elif isinstance(typ, enum.EnumMeta):
        return {typ}
    return functools.reduce(set.union, map(handle_type, get_args(typ)), set())