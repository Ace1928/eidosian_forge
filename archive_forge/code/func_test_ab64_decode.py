from __future__ import with_statement
from functools import partial
import warnings
from passlib.utils import is_ascii_safe, to_bytes
from passlib.utils.compat import irange, PY2, PY3, u, unicode, join_bytes, PYPY
from passlib.tests.utils import TestCase, hb, run_with_fixed_seeds
from passlib.utils.binary import h64, h64big
def test_ab64_decode(self):
    """ab64_decode()"""
    from passlib.utils.binary import ab64_decode
    self.assertEqual(ab64_decode(b'abc'), hb('69b7'))
    self.assertEqual(ab64_decode(u('abc')), hb('69b7'))
    self.assertRaises(ValueError, ab64_decode, u('abÿ'))
    self.assertRaises(TypeError, ab64_decode, b'ab\xff')
    self.assertRaises(TypeError, ab64_decode, b'ab!')
    self.assertRaises(TypeError, ab64_decode, u('ab!'))
    self.assertEqual(ab64_decode(b'abcd'), hb('69b71d'))
    self.assertRaises(ValueError, ab64_decode, b'abcde')
    self.assertEqual(ab64_decode(b'abcdef'), hb('69b71d79'))
    self.assertEqual(ab64_decode(b'abcdeQ'), hb('69b71d79'))
    self.assertEqual(ab64_decode(b'abcdefg'), hb('69b71d79f8'))
    self.assertEqual(ab64_decode(b'ab+/'), hb('69bfbf'))
    self.assertEqual(ab64_decode(b'ab./'), hb('69bfbf'))