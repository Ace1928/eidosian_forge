from dulwich import lru_cache
from dulwich.tests import TestCase
def test_by_usage(self):
    """Accessing entries bumps them up in priority."""
    cache = lru_cache.LRUCache(max_cache=2)
    cache['baz'] = 'biz'
    cache['foo'] = 'bar'
    self.assertEqual('biz', cache['baz'])
    cache['nub'] = 'in'
    self.assertNotIn('foo', cache)