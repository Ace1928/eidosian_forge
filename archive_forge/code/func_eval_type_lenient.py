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
def eval_type_lenient(value: Any, globalns: dict[str, Any] | None=None, localns: dict[str, Any] | None=None) -> Any:
    """Behaves like typing._eval_type, except it won't raise an error if a forward reference can't be resolved."""
    if value is None:
        value = NoneType
    elif isinstance(value, str):
        value = _make_forward_ref(value, is_argument=False, is_class=True)
    try:
        return eval_type_backport(value, globalns, localns)
    except NameError:
        return value