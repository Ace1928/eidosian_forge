import unittest
from cachetools import LFUCache
from . import CacheTestMixin
def test_lfu(self):
    cache = LFUCache(maxsize=2)
    cache[1] = 1
    cache[1]
    cache[2] = 2
    cache[3] = 3
    self.assertEqual(len(cache), 2)
    self.assertEqual(cache[1], 1)
    self.assertTrue(2 in cache or 3 in cache)
    self.assertTrue(2 not in cache or 3 not in cache)
    cache[4] = 4
    self.assertEqual(len(cache), 2)
    self.assertEqual(cache[4], 4)
    self.assertEqual(cache[1], 1)