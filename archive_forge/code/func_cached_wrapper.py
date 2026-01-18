from abc import abstractmethod
from abc import ABCMeta
import threading
import time
import uuid
def cached_wrapper(*args, **kwargs):
    try:
        key = (args, frozenset(kwargs.items())) if kwargs else args
    except TypeError as e:
        if self._ignore_unhashable_args:
            return func(*args, **kwargs)
        else:
            raise e
    else:
        val = cache.get(key, marker)
        if val is marker:
            val = func(*args, **kwargs)
            cache.put(key, val)
        return val