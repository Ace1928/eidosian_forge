import random
import time
import unittest
def test_ctor_no_size(self):
    from repoze.lru import UnboundedCache
    decorator = self._makeOne(maxsize=None)
    self.assertIsInstance(decorator.cache, UnboundedCache)
    self.assertEqual(decorator.cache._data, {})