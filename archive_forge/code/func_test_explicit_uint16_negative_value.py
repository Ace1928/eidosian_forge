import unittest
from .._nvlist import nvlist_in, nvlist_out, _lib
from ..ctypes import (
def test_explicit_uint16_negative_value(self):
    with self.assertRaises(OverflowError):
        props = {'key': uint16_t(-1)}
        self._dict_to_nvlist_to_dict(props)