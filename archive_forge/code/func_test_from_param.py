from ctypes import *
import unittest
import struct
def test_from_param(self):
    for t in signed_types + unsigned_types + float_types:
        self.assertEqual(ArgType, type(t.from_param(0)))