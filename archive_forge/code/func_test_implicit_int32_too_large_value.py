import unittest
from .._nvlist import nvlist_in, nvlist_out, _lib
from ..ctypes import (
def test_implicit_int32_too_large_value(self):
    with self.assertRaises(OverflowError):
        props = {'pool_context': 2 ** 31}
        self._dict_to_nvlist_to_dict(props)