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
def get_cls_type_hints_lenient(obj: Any, globalns: dict[str, Any] | None=None) -> dict[str, Any]:
    """Collect annotations from a class, including those from parent classes.

    Unlike `typing.get_type_hints`, this function will not error if a forward reference is not resolvable.
    """
    hints = {}
    for base in reversed(obj.__mro__):
        ann = base.__dict__.get('__annotations__')
        localns = dict(vars(base))
        if ann is not None and ann is not GetSetDescriptorType:
            for name, value in ann.items():
                hints[name] = eval_type_lenient(value, globalns, localns)
    return hints