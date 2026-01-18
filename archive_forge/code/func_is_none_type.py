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
def is_none_type(type_: Any) -> bool:
    return type_ in NONE_TYPES