import random
import time
import unittest
def test_ctor_w_size_no_timeout(self):
    from repoze.lru import LRUCache
    decorator = self._makeOne(maxsize=10)
    self.assertIsInstance(decorator.cache, LRUCache)
    self.assertEqual(decorator.cache.size, 10)