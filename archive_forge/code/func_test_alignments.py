from ctypes import *
import unittest
import struct
def test_alignments(self):
    for t in signed_types + unsigned_types + float_types:
        code = t._type_
        align = struct.calcsize('c%c' % code) - struct.calcsize(code)
        self.assertEqual((code, alignment(t)), (code, align))
        self.assertEqual((code, alignment(t())), (code, align))