import unittest
from .._nvlist import nvlist_in, nvlist_out, _lib
from ..ctypes import (
def test_explicit_boolean_array(self):
    props = {'key': [boolean_t(False), boolean_t(True)]}
    res = self._dict_to_nvlist_to_dict(props)
    self._assertIntArrayDictsEqual(props, res)