import collections
import functools
import math
import random
import time
from . import FIFOCache, LFUCache, LRUCache, MRUCache, RRCache, TTLCache
from . import keys
def cache_clear():
    with lock:
        try:
            cache.clear()
        finally:
            stats[:] = [0, 0]