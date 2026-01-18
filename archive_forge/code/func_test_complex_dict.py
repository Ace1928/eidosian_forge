import unittest
from .._nvlist import nvlist_in, nvlist_out, _lib
from ..ctypes import (
def test_complex_dict(self):
    props = {'key1': 'str', 'key2': 10, 'key3': {'skey1': True, 'skey2': None, 'skey3': [True, False, True]}, 'key4': ['ab', 'bc'], 'key5': [2 ** 64 - 1, 1, 2, 3], 'key6': [{'skey71': 'a', 'skey72': 'b'}, {'skey71': 'c', 'skey72': 'd'}, {'skey71': 'e', 'skey72': 'f'}], 'type': 2 ** 32 - 1, 'pool_context': -2 ** 31}
    res = self._dict_to_nvlist_to_dict(props)
    self.assertEqual(props, res)