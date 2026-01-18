import math
import sys
import unittest
from llvmlite.ir import (
from llvmlite.tests import TestCase
@unittest.skipUnless(PY36_OR_LATER, 'py36+ only')
def test_half_repr(self):

    def check_repr(val, expected):
        c = Constant(HalfType(), val)
        self.assertEqual(str(c), expected)
    check_repr(math.pi, 'half 0x4009200000000000')
    check_repr(float('inf'), 'half 0x7ff0000000000000')
    check_repr(float('-inf'), 'half 0xfff0000000000000')