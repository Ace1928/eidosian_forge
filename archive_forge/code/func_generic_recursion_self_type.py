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
@contextmanager
def generic_recursion_self_type(origin: type[BaseModel], args: tuple[Any, ...]) -> Iterator[PydanticRecursiveRef | None]:
    """This contextmanager should be placed around the recursive calls used to build a generic type,
    and accept as arguments the generic origin type and the type arguments being passed to it.

    If the same origin and arguments are observed twice, it implies that a self-reference placeholder
    can be used while building the core schema, and will produce a schema_ref that will be valid in the
    final parent schema.
    """
    previously_seen_type_refs = _generic_recursion_cache.get()
    if previously_seen_type_refs is None:
        previously_seen_type_refs = set()
        token = _generic_recursion_cache.set(previously_seen_type_refs)
    else:
        token = None
    try:
        type_ref = get_type_ref(origin, args_override=args)
        if type_ref in previously_seen_type_refs:
            self_type = PydanticRecursiveRef(type_ref=type_ref)
            yield self_type
        else:
            previously_seen_type_refs.add(type_ref)
            yield None
    finally:
        if token:
            _generic_recursion_cache.reset(token)