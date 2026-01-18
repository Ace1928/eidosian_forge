import sys
from test import support
import unittest
from ctypes import *
from ctypes.test import need_symbol
def test_memset(self):
    a = create_string_buffer(1000000)
    result = memset(a, ord('x'), 16)
    self.assertEqual(a.value, b'xxxxxxxxxxxxxxxx')
    self.assertEqual(string_at(result), b'xxxxxxxxxxxxxxxx')
    self.assertEqual(string_at(a), b'xxxxxxxxxxxxxxxx')
    self.assertEqual(string_at(a, 20), b'xxxxxxxxxxxxxxxx\x00\x00\x00\x00')