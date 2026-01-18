import collections
import functools
import math
import random
import time
from . import FIFOCache, LFUCache, LRUCache, MRUCache, RRCache, TTLCache
from . import keys
def cache_info():
    with lock:
        hits, misses = stats
        maxsize = cache.maxsize
        currsize = cache.currsize
    return _CacheInfo(hits, misses, maxsize, currsize)