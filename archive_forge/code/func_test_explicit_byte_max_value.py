import unittest
from .._nvlist import nvlist_in, nvlist_out, _lib
from ..ctypes import (
def test_explicit_byte_max_value(self):
    props = {'key': uchar_t(2 ** 8 - 1)}
    res = self._dict_to_nvlist_to_dict(props)
    self._assertIntDictsEqual(props, res)