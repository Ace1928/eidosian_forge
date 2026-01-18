from __future__ import annotations
import asyncio
import inspect
import math
import operator
from collections.abc import Iterable, Iterator
from functools import partial
from types import FunctionType, MethodType
from typing import Any, Callable, Optional
from .depends import depends
from .display import _display_accessors, _reactive_display_objs
from .parameterized import (
from .parameters import Boolean, Event
from ._utils import _to_async_gen, iscoroutinefunction, full_groupby
@classmethod
def register_accessor(cls, name: str, accessor: Callable[[rx], Any], predicate: Optional[Callable[[Any], bool]]=None):
    """
        Registers an accessor that extends rx with custom behavior.

        Arguments
        ---------
        name: str
          The name of the accessor will be attribute-accessible under.
        accessor: Callable[[rx], any]
          A callable that will return the accessor namespace object
          given the rx object it is registered on.
        predicate: Callable[[Any], bool] | None
        """
    cls._accessors[name] = (accessor, predicate)