from dulwich import lru_cache
from dulwich.tests import TestCase
def test_no_add_over_size(self):
    """Adding a large value may not be cached at all."""
    cache = lru_cache.LRUSizeCache(max_size=10, after_cleanup_size=5)
    self.assertEqual(0, cache._value_size)
    self.assertEqual({}, cache.items())
    cache.add('test', 'key')
    self.assertEqual(3, cache._value_size)
    self.assertEqual({'test': 'key'}, cache.items())
    cache.add('test2', 'key that is too big')
    self.assertEqual(3, cache._value_size)
    self.assertEqual({'test': 'key'}, cache.items())
    cache.add('test3', 'bigkey')
    self.assertEqual(3, cache._value_size)
    self.assertEqual({'test': 'key'}, cache.items())
    cache.add('test4', 'bikey')
    self.assertEqual(3, cache._value_size)
    self.assertEqual({'test': 'key'}, cache.items())