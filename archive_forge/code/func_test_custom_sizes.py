from dulwich import lru_cache
from dulwich.tests import TestCase
def test_custom_sizes(self):

    def size_of_list(lst):
        return sum((len(x) for x in lst))
    cache = lru_cache.LRUSizeCache(max_size=20, after_cleanup_size=10, compute_size=size_of_list)
    cache.add('key1', ['val', 'ue'])
    cache.add('key2', ['val', 'ue2'])
    cache.add('key3', ['val', 'ue23'])
    self.assertEqual(5 + 6 + 7, cache._value_size)
    cache['key2']
    cache.add('key4', ['value', '234'])
    self.assertEqual(8, cache._value_size)
    self.assertEqual({'key4': ['value', '234']}, cache.items())