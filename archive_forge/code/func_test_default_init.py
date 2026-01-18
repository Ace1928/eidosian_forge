from ctypes import *
import unittest
import struct
def test_default_init(self):
    for t in signed_types + unsigned_types + float_types:
        self.assertEqual(t().value, 0)