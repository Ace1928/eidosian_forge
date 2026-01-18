from collections import namedtuple
from functools import partial
from threading import local
from .promise import Promise, async_instance, get_default_scheduler
def prime(self, key, value):
    """
        Adds the provied key and value to the cache. If the key already exists, no
        change is made. Returns itself for method chaining.
        """
    cache_key = self.get_cache_key(key)
    if cache_key not in self._promise_cache:
        if isinstance(value, Exception):
            promise = Promise.reject(value)
        else:
            promise = Promise.resolve(value)
        self._promise_cache[cache_key] = promise
    return self