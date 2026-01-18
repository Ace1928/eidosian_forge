import struct
import unittest
import zope.interface  # noqa: try to load a C module for side effects
def test_no_fast_math_optimization(self):
    zero_bits = struct.unpack('!Q', struct.pack('!d', 0.0))[0]
    next_up = zero_bits + 1
    smallest_subnormal = struct.unpack('!d', struct.pack('!Q', next_up))[0]
    self.assertNotEqual(smallest_subnormal, 0.0)