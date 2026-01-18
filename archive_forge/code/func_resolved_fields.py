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
@_unsafe_cache.unsafe_cache(maxsize=1024)
def resolved_fields(cls: TypeForm) -> List[dataclasses.Field]:
    """Similar to dataclasses.fields(), but includes dataclasses.InitVar types and
    resolves forward references."""
    assert dataclasses.is_dataclass(cls)
    fields = []
    annotations = get_type_hints(cast(Callable, cls), include_extras=True)
    for field in getattr(cls, '__dataclass_fields__').values():
        field = copy.copy(field)
        field.type = annotations[field.name]
        if get_origin(field.type) is ClassVar:
            continue
        if isinstance(field.type, dataclasses.InitVar):
            field.type = field.type.type
        fields.append(field)
    return fields