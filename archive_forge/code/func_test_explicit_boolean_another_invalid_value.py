import unittest
from .._nvlist import nvlist_in, nvlist_out, _lib
from ..ctypes import (
def test_explicit_boolean_another_invalid_value(self):
    with self.assertRaises(OverflowError):
        props = {'key': boolean_t(-1)}
        self._dict_to_nvlist_to_dict(props)