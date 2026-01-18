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
def get_cls_types_namespace(cls: type[Any], parent_namespace: dict[str, Any] | None=None) -> dict[str, Any]:
    ns = add_module_globals(cls, parent_namespace)
    ns[cls.__name__] = cls
    return ns