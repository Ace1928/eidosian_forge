import unittest
from .._nvlist import nvlist_in, nvlist_out, _lib
from ..ctypes import (
def test_explict_int16_array_too_small_value(self):
    with self.assertRaises(OverflowError):
        props = {'key': [int16_t(0), int16_t(-2 ** 15 - 1)]}
        self._dict_to_nvlist_to_dict(props)