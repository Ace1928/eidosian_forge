import unittest
from .._nvlist import nvlist_in, nvlist_out, _lib
from ..ctypes import (
def test_explict_int32_array(self):
    props = {'key': [int32_t(0), int32_t(1), int32_t(2 ** 31 - 1), int32_t(-2 ** 31)]}
    res = self._dict_to_nvlist_to_dict(props)
    self._assertIntArrayDictsEqual(props, res)