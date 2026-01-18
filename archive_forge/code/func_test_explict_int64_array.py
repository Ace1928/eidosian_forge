import unittest
from .._nvlist import nvlist_in, nvlist_out, _lib
from ..ctypes import (
def test_explict_int64_array(self):
    props = {'key': [int64_t(0), int64_t(1), int64_t(2 ** 63 - 1), int64_t(-2 ** 63)]}
    res = self._dict_to_nvlist_to_dict(props)
    self._assertIntArrayDictsEqual(props, res)