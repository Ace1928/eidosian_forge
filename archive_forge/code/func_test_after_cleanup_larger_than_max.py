from dulwich import lru_cache
from dulwich.tests import TestCase
def test_after_cleanup_larger_than_max(self):
    cache = lru_cache.LRUCache(max_cache=5, after_cleanup_count=10)
    self.assertEqual(5, cache._after_cleanup_count)