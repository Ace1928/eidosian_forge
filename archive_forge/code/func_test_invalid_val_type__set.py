import unittest
from .._nvlist import nvlist_in, nvlist_out, _lib
from ..ctypes import (
def test_invalid_val_type__set(self):
    with self.assertRaises(TypeError):
        self._dict_to_nvlist_to_dict({'key': set(1, 2)})