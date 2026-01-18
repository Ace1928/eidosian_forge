import unittest
from cachetools import MRUCache
from . import CacheTestMixin
def test_evict__with_access(self):
    cache = MRUCache(maxsize=2)
    cache[1] = 1
    cache[2] = 2
    cache[1]
    cache[2]
    cache[3] = 3
    assert 2 not in cache, "Wrong key was evicted. Should have been '2'."
    assert 1 in cache
    assert 3 in cache