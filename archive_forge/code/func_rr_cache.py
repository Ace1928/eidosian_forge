import collections
import functools
import math
import random
import time
from . import FIFOCache, LFUCache, LRUCache, MRUCache, RRCache, TTLCache
from . import keys
def rr_cache(maxsize=128, choice=random.choice, typed=False):
    """Decorator to wrap a function with a memoizing callable that saves
    up to `maxsize` results based on a Random Replacement (RR)
    algorithm.

    """
    if maxsize is None:
        return _cache(_UnboundCache(), typed)
    elif callable(maxsize):
        return _cache(RRCache(128, choice), typed)(maxsize)
    else:
        return _cache(RRCache(maxsize, choice), typed)