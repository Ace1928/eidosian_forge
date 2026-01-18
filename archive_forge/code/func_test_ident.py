from numba import int32, int64, uint32, uint64, float32, float64
from numba.core.types import range_iter32_type
from numba.core import itanium_mangler
import unittest
def test_ident(self):
    got = itanium_mangler.mangle_identifier('apple')
    expect = '5apple'
    self.assertEqual(expect, got)
    got = itanium_mangler.mangle_identifier('ap_ple')
    expect = '6ap_ple'
    self.assertEqual(expect, got)
    got = itanium_mangler.mangle_identifier('apple213')
    expect = '8apple213'
    self.assertEqual(expect, got)