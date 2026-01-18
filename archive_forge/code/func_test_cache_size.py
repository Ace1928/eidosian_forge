from dulwich import lru_cache
from dulwich.tests import TestCase
def test_cache_size(self):
    cache = lru_cache.LRUCache(max_cache=10)
    self.assertEqual(10, cache.cache_size())
    cache = lru_cache.LRUCache(max_cache=256)
    self.assertEqual(256, cache.cache_size())
    cache.resize(512)
    self.assertEqual(512, cache.cache_size())