import sys
import unittest
from Cython.Utils import (
def test_cache_method_name(self):
    method_name = 'foo'
    cache_name = _build_cache_name(method_name)
    match = _CACHE_NAME_PATTERN.match(cache_name)
    self.assertIsNot(match, None)
    self.assertEqual(match.group(1), method_name)