from ctypes import *
from ctypes.test import need_symbol
import unittest
@need_symbol('c_wchar')
def test_create_unicode_buffer_non_bmp(self):
    expected = 5 if sizeof(c_wchar) == 2 else 3
    for s in ('𐀀\U00100000', '𐀀\U0010ffff'):
        b = create_unicode_buffer(s)
        self.assertEqual(len(b), expected)
        self.assertEqual(b[-1], '\x00')