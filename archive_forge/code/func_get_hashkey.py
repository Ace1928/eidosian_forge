import operator
import unittest
from cachetools import LRUCache, cachedmethod, keys
@cachedmethod(operator.attrgetter('cache'), key=keys.hashkey)
def get_hashkey(self, value):
    return value