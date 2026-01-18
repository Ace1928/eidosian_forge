from __future__ import absolute_import, division, print_function
import logging
from functools import wraps, update_wrapper
import types
from warnings import warn
from passlib.utils.compat import PY3
def memoize_single_value(func):
    """
    decorator for function which takes no args,
    and memoizes result.  exposes a ``.clear_cache`` method
    to clear the cached value.
    """
    cache = {}

    @wraps(func)
    def wrapper():
        try:
            return cache[True]
        except KeyError:
            pass
        value = cache[True] = func()
        return value

    def clear_cache():
        cache.pop(True, None)
    wrapper.clear_cache = clear_cache
    return wrapper