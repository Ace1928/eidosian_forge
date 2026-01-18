import math
import unittest
from cachetools import TLRUCache
from . import CacheTestMixin
def default_ttu(_key, _value, _time):
    return math.inf