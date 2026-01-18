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
@classmethod
def promisify(cls, f):
    if not callable(f):
        warn('Promise.promisify is now a function decorator, please use Promise.resolve instead.')
        return cls.resolve(f)

    @wraps(f)
    def wrapper(*args, **kwargs):

        def executor(resolve, reject):
            return resolve(f(*args, **kwargs))
        return cls(executor)
    return wrapper