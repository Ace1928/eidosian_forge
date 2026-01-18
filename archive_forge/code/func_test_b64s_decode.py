from __future__ import with_statement
from functools import partial
import warnings
from passlib.utils import is_ascii_safe, to_bytes
from passlib.utils.compat import irange, PY2, PY3, u, unicode, join_bytes, PYPY
from passlib.tests.utils import TestCase, hb, run_with_fixed_seeds
from passlib.utils.binary import h64, h64big
def test_b64s_decode(self):
    """b64s_decode()"""
    from passlib.utils.binary import b64s_decode
    self.assertEqual(b64s_decode(b'abc'), hb('69b7'))
    self.assertEqual(b64s_decode(u('abc')), hb('69b7'))
    self.assertRaises(ValueError, b64s_decode, u('abÿ'))
    self.assertRaises(TypeError, b64s_decode, b'ab\xff')
    self.assertRaises(TypeError, b64s_decode, b'ab!')
    self.assertRaises(TypeError, b64s_decode, u('ab!'))
    self.assertEqual(b64s_decode(b'abcd'), hb('69b71d'))
    self.assertRaises(ValueError, b64s_decode, b'abcde')
    self.assertEqual(b64s_decode(b'abcdef'), hb('69b71d79'))
    self.assertEqual(b64s_decode(b'abcdeQ'), hb('69b71d79'))
    self.assertEqual(b64s_decode(b'abcdefg'), hb('69b71d79f8'))