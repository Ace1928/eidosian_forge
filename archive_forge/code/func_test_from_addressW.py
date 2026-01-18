import unittest
from test.support import bigmemtest, _2G
import sys
from ctypes import *
from ctypes.test import need_symbol
@need_symbol('create_unicode_buffer')
def test_from_addressW(self):
    p = create_unicode_buffer('foo')
    sz = (c_wchar * 3).from_address(addressof(p))
    self.assertEqual(sz[:], 'foo')
    self.assertEqual(sz[:], 'foo')
    self.assertEqual(sz[::-1], 'oof')
    self.assertEqual(sz[::3], 'f')
    self.assertEqual(sz[1:4:2], 'o')
    self.assertEqual(sz.value, 'foo')