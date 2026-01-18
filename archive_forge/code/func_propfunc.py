from importlib import import_module
from typing import Callable
from functools import lru_cache, wraps
def propfunc(self):
    val = getattr(self, attrname, _cached_property_sentinel)
    if val is _cached_property_sentinel:
        val = func(self)
        setattr(self, attrname, val)
    return val