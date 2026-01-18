import sys
from test import support
import unittest
from ctypes import *
from ctypes.test import need_symbol
def test_memmove(self):
    a = create_string_buffer(1000000)
    p = b'Hello, World'
    result = memmove(a, p, len(p))
    self.assertEqual(a.value, b'Hello, World')
    self.assertEqual(string_at(result), b'Hello, World')
    self.assertEqual(string_at(result, 5), b'Hello')
    self.assertEqual(string_at(result, 16), b'Hello, World\x00\x00\x00\x00')
    self.assertEqual(string_at(result, 0), b'')