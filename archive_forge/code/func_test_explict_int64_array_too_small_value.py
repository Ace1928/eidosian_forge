import unittest
from .._nvlist import nvlist_in, nvlist_out, _lib
from ..ctypes import (
def test_explict_int64_array_too_small_value(self):
    with self.assertRaises(OverflowError):
        props = {'key': [int64_t(0), int64_t(-2 ** 63 - 1)]}
        self._dict_to_nvlist_to_dict(props)