from ctypes import *
import unittest
import struct
def test_bool_values(self):
    from operator import truth
    for t, v in zip(bool_types, bool_values):
        self.assertEqual(t(v).value, truth(v))