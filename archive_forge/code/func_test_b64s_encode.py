from __future__ import with_statement
from functools import partial
import warnings
from passlib.utils import is_ascii_safe, to_bytes
from passlib.utils.compat import irange, PY2, PY3, u, unicode, join_bytes, PYPY
from passlib.tests.utils import TestCase, hb, run_with_fixed_seeds
from passlib.utils.binary import h64, h64big
def test_b64s_encode(self):
    """b64s_encode()"""
    from passlib.utils.binary import b64s_encode
    self.assertEqual(b64s_encode(hb('69b7')), b'abc')
    self.assertRaises(TypeError if PY3 else UnicodeEncodeError, b64s_encode, hb('69b7').decode('latin-1'))
    self.assertEqual(b64s_encode(hb('69b71d')), b'abcd')
    self.assertEqual(b64s_encode(hb('69b71d79')), b'abcdeQ')
    self.assertEqual(b64s_encode(hb('69b71d79f8')), b'abcdefg')
    self.assertEqual(b64s_encode(hb('69bfbf')), b'ab+/')