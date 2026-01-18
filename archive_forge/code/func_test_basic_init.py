from dulwich import lru_cache
from dulwich.tests import TestCase
def test_basic_init(self):
    cache = lru_cache.LRUSizeCache()
    self.assertEqual(2048, cache._max_cache)
    self.assertEqual(int(cache._max_size * 0.8), cache._after_cleanup_size)
    self.assertEqual(0, cache._value_size)