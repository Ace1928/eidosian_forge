import random
import time
import unittest
def test_multiargs_keywords_ignore_unhashable_true(self):
    cache = DummyLRUCache()
    decorator = self._makeOne(0, cache, ignore_unhashable_args=True)

    def moreargs(*args, **kwargs):
        return (args, kwargs)
    decorated = decorator(moreargs)
    result = decorated(3, 4, 5, a=1, b=[1, 2, 3])
    self.assertEqual(len(cache), 0)
    self.assertEqual(result, ((3, 4, 5), {'a': 1, 'b': [1, 2, 3]}))