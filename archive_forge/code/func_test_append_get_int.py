import ctypes
import struct
from numba.tests.support import TestCase
from numba import _helperlib
def test_append_get_int(self):
    l = List(self, 8, 1)
    l.append(struct.pack('q', 1))
    self.assertEqual(len(l), 1)
    r = struct.unpack('q', l[0])[0]
    self.assertEqual(r, 1)