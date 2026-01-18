import collections
import functools
import math
import random
import time
from . import FIFOCache, LFUCache, LRUCache, MRUCache, RRCache, TTLCache
from . import keys
def fifo_cache(maxsize=128, typed=False):
    """Decorator to wrap a function with a memoizing callable that saves
    up to `maxsize` results based on a First In First Out (FIFO)
    algorithm.

    """
    if maxsize is None:
        return _cache(_UnboundCache(), typed)
    elif callable(maxsize):
        return _cache(FIFOCache(128), typed)(maxsize)
    else:
        return _cache(FIFOCache(maxsize), typed)