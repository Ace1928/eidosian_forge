import random
import time
import unittest
def test_eviction_counter(self):
    cache = self._makeOne(2)
    cache.put(1, 1)
    cache.put(2, 1)
    self.assertEqual(cache.evictions, 0)
    cache.put(3, 1)
    cache.put(4, 1)
    self.assertEqual(cache.evictions, 2)
    cache.put(3, 1)
    cache.put(4, 1)
    self.assertEqual(cache.evictions, 2)
    cache.clear()
    self.assertEqual(cache.evictions, 0)