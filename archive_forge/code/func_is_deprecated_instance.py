from __future__ import annotations as _annotations
import dataclasses
import re
import sys
import types
import typing
import warnings
from collections.abc import Callable
from functools import partial
from types import GetSetDescriptorType
from typing import TYPE_CHECKING, Any, Final
from typing_extensions import Annotated, Literal, TypeAliasType, TypeGuard, deprecated, get_args, get_origin
def is_deprecated_instance(instance: Any) -> TypeGuard[deprecated]:
    return isinstance(instance, DEPRECATED_TYPES)