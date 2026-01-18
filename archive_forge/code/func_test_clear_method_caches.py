import sys
import unittest
from Cython.Utils import (
def test_clear_method_caches(self):
    obj = Cached()
    value = iter(range(3))
    cache = {(value,): 1}
    obj.cached_next(value)
    clear_method_caches(obj)
    self.set_of_names_equal(obj, set())
    self.assertEqual(obj.cached_next(value), 1)
    self.set_of_names_equal(obj, {NAMES})
    self.assertEqual(getattr(obj, CACHE_NAME), cache)