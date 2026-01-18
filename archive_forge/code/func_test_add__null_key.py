from dulwich import lru_cache
from dulwich.tests import TestCase
def test_add__null_key(self):
    cache = lru_cache.LRUSizeCache()
    self.assertRaises(ValueError, cache.add, lru_cache._null_key, 1)