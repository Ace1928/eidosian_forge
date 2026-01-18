from collections import namedtuple
from functools import partial, wraps
from sys import version_info, exc_info
from threading import RLock
from types import TracebackType
from weakref import WeakKeyDictionary
from .async_ import Async
from .compat import (
from .utils import deprecated, integer_types, string_types, text_type, binary_type, warn
from .promise_list import PromiseList
from .schedulers.immediate import ImmediateScheduler
from typing import TypeVar, Generic
def is_future_like(_type):
    if _type not in _type_done_callbacks:
        _type_done_callbacks[_type] = callable(getattr(_type, 'add_done_callback', None))
    return _type_done_callbacks[_type]