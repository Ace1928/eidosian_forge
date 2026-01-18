from __future__ import annotations as _annotations
import dataclasses
import sys
import types
import typing
from collections.abc import Callable
from functools import partial
from types import GetSetDescriptorType
from typing import TYPE_CHECKING, Any, Final
from typing_extensions import Annotated, Literal, TypeAliasType, TypeGuard, get_args, get_origin
def parent_frame_namespace(*, parent_depth: int=2) -> dict[str, Any] | None:
    """We allow use of items in parent namespace to get around the issue with `get_type_hints` only looking in the
    global module namespace. See https://github.com/pydantic/pydantic/issues/2678#issuecomment-1008139014 -> Scope
    and suggestion at the end of the next comment by @gvanrossum.

    WARNING 1: it matters exactly where this is called. By default, this function will build a namespace from the
    parent of where it is called.

    WARNING 2: this only looks in the parent namespace, not other parents since (AFAIK) there's no way to collect a
    dict of exactly what's in scope. Using `f_back` would work sometimes but would be very wrong and confusing in many
    other cases. See https://discuss.python.org/t/is-there-a-way-to-access-parent-nested-namespaces/20659.
    """
    frame = sys._getframe(parent_depth)
    if frame.f_back is None:
        return None
    else:
        return frame.f_locals