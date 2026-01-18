from __future__ import annotations
import logging # isort:skip
from copy import copy
from types import FunctionType
from typing import (
from ...util.deprecation import deprecated
from .singletons import Undefined
from .wrappers import PropertyValueColumnData, PropertyValueContainer
@classmethod
def is_unstable(cls, value: Any) -> TypeGuard[Callable[[], Any]]:
    from .instance import InstanceDefault
    return isinstance(value, (FunctionType, InstanceDefault))