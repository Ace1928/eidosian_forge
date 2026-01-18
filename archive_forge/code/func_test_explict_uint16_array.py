import unittest
from .._nvlist import nvlist_in, nvlist_out, _lib
from ..ctypes import (
def test_explict_uint16_array(self):
    props = {'key': [uint16_t(0), uint16_t(1), uint16_t(2 ** 16 - 1)]}
    res = self._dict_to_nvlist_to_dict(props)
    self._assertIntArrayDictsEqual(props, res)