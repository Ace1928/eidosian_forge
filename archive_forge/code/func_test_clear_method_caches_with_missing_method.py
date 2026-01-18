import sys
import unittest
from Cython.Utils import (
def test_clear_method_caches_with_missing_method(self):
    obj = Cached()
    method_name = 'bar'
    cache_name = _build_cache_name(method_name)
    names = (cache_name, method_name)
    setattr(obj, cache_name, object())
    self.assertFalse(hasattr(obj, method_name))
    self.set_of_names_equal(obj, {names})
    clear_method_caches(obj)
    self.set_of_names_equal(obj, {names})