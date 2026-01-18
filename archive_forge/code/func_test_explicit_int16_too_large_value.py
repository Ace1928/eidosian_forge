import unittest
from .._nvlist import nvlist_in, nvlist_out, _lib
from ..ctypes import (
def test_explicit_int16_too_large_value(self):
    with self.assertRaises(OverflowError):
        props = {'key': int16_t(2 ** 15)}
        self._dict_to_nvlist_to_dict(props)