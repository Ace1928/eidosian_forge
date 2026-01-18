from __future__ import unicode_literals
from collections import deque
from functools import wraps
def memoized(maxsize=1024):
    """
    Momoization decorator for immutable classes and pure functions.
    """
    cache = SimpleCache(maxsize=maxsize)

    def decorator(obj):

        @wraps(obj)
        def new_callable(*a, **kw):

            def create_new():
                return obj(*a, **kw)
            key = (a, tuple(kw.items()))
            return cache.get(key, create_new)
        return new_callable
    return decorator