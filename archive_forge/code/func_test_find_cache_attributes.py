import sys
import unittest
from Cython.Utils import (
def test_find_cache_attributes(self):
    obj = Cached()
    method_name = 'bar'
    cache_name = _build_cache_name(method_name)
    setattr(obj, CACHE_NAME, {})
    setattr(obj, cache_name, {})
    self.assertFalse(hasattr(obj, method_name))
    self.set_of_names_equal(obj, {NAMES, (cache_name, method_name)})