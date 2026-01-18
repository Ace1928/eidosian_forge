from dulwich import lru_cache
from dulwich.tests import TestCase
def test_cleanup_2(self):
    cache = lru_cache.LRUCache(max_cache=5, after_cleanup_count=2)
    cache.add(1, 10)
    cache.add(2, 20)
    cache.add(3, 25)
    cache.add(4, 30)
    cache.add(5, 35)
    self.assertEqual(5, len(cache))
    cache.cleanup()
    self.assertEqual(2, len(cache))