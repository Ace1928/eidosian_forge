import random
import time
import unittest
def test_expiring(self):
    size = 10
    timeout = 10
    name = 'name'
    cache = self._makeOne(maxsize=size, timeout=timeout)
    for i in range(100):
        if not i:
            decorator = cache.expiring_lrucache(name=name)
            decorated = decorator(_adder)
            self.assertEqual(cache._cache[name].size, size)
        else:
            decorator = cache.expiring_lrucache()
            decorated = decorator(_adder)
            self.assertEqual(decorator.cache.default_timeout, timeout)
        decorated(10)
    self.assertEqual(len(cache._cache), 100)
    for _cache in cache._cache.values():
        self.assertEqual(_cache.size, size)
        self.assertEqual(_cache.default_timeout, timeout)
        self.assertEqual(len(_cache.data), 1)
    cache.clear()
    for _cache in cache._cache.values():
        self.assertEqual(_cache.size, size)
        self.assertEqual(len(_cache.data), 0)