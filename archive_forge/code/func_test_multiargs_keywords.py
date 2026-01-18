import random
import time
import unittest
def test_multiargs_keywords(self):
    cache = DummyLRUCache()
    decorator = self._makeOne(0, cache)

    def moreargs(*args, **kwargs):
        return (args, kwargs)
    decorated = decorator(moreargs)
    result = decorated(3, 4, 5, a=1, b=2, c=3)
    self.assertEqual(cache[(3, 4, 5), frozenset([('a', 1), ('b', 2), ('c', 3)])], ((3, 4, 5), {'a': 1, 'b': 2, 'c': 3}))
    self.assertEqual(result, ((3, 4, 5), {'a': 1, 'b': 2, 'c': 3}))
    self.assertEqual(len(cache), 1)