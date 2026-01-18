import collections.abc
import gc
import inspect
import re
import sys
import weakref
from functools import partial, wraps
from itertools import chain
from typing import (
from scrapy.utils.asyncgen import as_async_generator
def memoizemethod_noargs(method: Callable) -> Callable:
    """Decorator to cache the result of a method (without arguments) using a
    weak reference to its object
    """
    cache: weakref.WeakKeyDictionary[Any, Any] = weakref.WeakKeyDictionary()

    @wraps(method)
    def new_method(self: Any, *args: Any, **kwargs: Any) -> Any:
        if self not in cache:
            cache[self] = method(self, *args, **kwargs)
        return cache[self]
    return new_method