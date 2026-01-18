import unittest
from .._nvlist import nvlist_in, nvlist_out, _lib
from ..ctypes import (
def test_explicit_int32_too_small_value(self):
    with self.assertRaises(OverflowError):
        props = {'key': int32_t(-2 ** 31 - 1)}
        self._dict_to_nvlist_to_dict(props)