import unittest
from .._nvlist import nvlist_in, nvlist_out, _lib
from ..ctypes import (
def test_explict_byte_array(self):
    props = {'key': [uchar_t(0), uchar_t(1), uchar_t(2 ** 8 - 1)]}
    res = self._dict_to_nvlist_to_dict(props)
    self._assertIntArrayDictsEqual(props, res)