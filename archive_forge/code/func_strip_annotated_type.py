from __future__ import annotations
from typing import Any, TypeVar, Iterable, cast
from collections import abc as _c_abc
from typing_extensions import Required, Annotated, get_args, get_origin
from .._types import InheritsGeneric
from .._compat import is_union as _is_union
def strip_annotated_type(typ: type) -> type:
    if is_required_type(typ) or is_annotated_type(typ):
        return strip_annotated_type(cast(type, get_args(typ)[0]))
    return typ