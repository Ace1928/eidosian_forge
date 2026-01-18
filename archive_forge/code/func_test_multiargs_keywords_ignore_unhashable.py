import random
import time
import unittest
def test_multiargs_keywords_ignore_unhashable(self):
    cache = DummyLRUCache()
    decorator = self._makeOne(0, cache, ignore_unhashable_args=False)

    def moreargs(*args, **kwargs):
        return (args, kwargs)
    decorated = decorator(moreargs)
    with self.assertRaises(TypeError):
        decorated(3, 4, 5, a=1, b=[1, 2, 3])