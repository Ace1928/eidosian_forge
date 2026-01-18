import unittest
from .._nvlist import nvlist_in, nvlist_out, _lib
from ..ctypes import (
def test_explicit_int8_max_value(self):
    props = {'key': int8_t(2 ** 7 - 1)}
    res = self._dict_to_nvlist_to_dict(props)
    self._assertIntDictsEqual(props, res)