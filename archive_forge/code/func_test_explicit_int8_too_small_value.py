import unittest
from .._nvlist import nvlist_in, nvlist_out, _lib
from ..ctypes import (
def test_explicit_int8_too_small_value(self):
    with self.assertRaises(OverflowError):
        props = {'key': int8_t(-2 ** 7 - 1)}
        self._dict_to_nvlist_to_dict(props)