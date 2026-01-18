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
def is_self_type(tp: Any) -> bool:
    """Check if a given class is a Self type (from `typing` or `typing_extensions`)"""
    return isinstance(tp, typing_base) and getattr(tp, '_name', None) == 'Self'