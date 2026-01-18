import unittest
from .._nvlist import nvlist_in, nvlist_out, _lib
from ..ctypes import (
def test_invalid_array_val_type(self):
    with self.assertRaises(TypeError):
        self._dict_to_nvlist_to_dict({'key': [(1, 2), (3, 4)]})