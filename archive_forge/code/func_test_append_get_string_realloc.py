import ctypes
import struct
from numba.tests.support import TestCase
from numba import _helperlib
def test_append_get_string_realloc(self):
    l = List(self, 8, 1)
    l.append(b'abcdefgh')
    self.assertEqual(len(l), 1)
    l.append(b'hijklmno')
    self.assertEqual(len(l), 2)
    r = l[1]
    self.assertEqual(r, b'hijklmno')