from dulwich import lru_cache
from dulwich.tests import TestCase
def test_no_add_over_size_cleanup(self):
    """If a large value is not cached, we will call cleanup right away."""
    cleanup_calls = []

    def cleanup(key, value):
        cleanup_calls.append((key, value))
    cache = lru_cache.LRUSizeCache(max_size=10, after_cleanup_size=5)
    self.assertEqual(0, cache._value_size)
    self.assertEqual({}, cache.items())
    cache.add('test', 'key that is too big', cleanup=cleanup)
    self.assertEqual(0, cache._value_size)
    self.assertEqual({}, cache.items())
    self.assertEqual([('test', 'key that is too big')], cleanup_calls)