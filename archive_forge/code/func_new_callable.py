from __future__ import unicode_literals
from collections import deque
from functools import wraps
@wraps(obj)
def new_callable(*a, **kw):

    def create_new():
        return obj(*a, **kw)
    key = (a, tuple(kw.items()))
    return cache.get(key, create_new)