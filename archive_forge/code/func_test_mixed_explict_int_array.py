import unittest
from .._nvlist import nvlist_in, nvlist_out, _lib
from ..ctypes import (
def test_mixed_explict_int_array(self):
    with self.assertRaises(TypeError):
        props = {'key': [uint64_t(0), uint32_t(0)]}
        self._dict_to_nvlist_to_dict(props)