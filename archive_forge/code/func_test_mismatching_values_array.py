import unittest
from .._nvlist import nvlist_in, nvlist_out, _lib
from ..ctypes import (
def test_mismatching_values_array(self):
    props = {'key': [1, 'string']}
    with self.assertRaises(TypeError):
        self._dict_to_nvlist_to_dict(props)