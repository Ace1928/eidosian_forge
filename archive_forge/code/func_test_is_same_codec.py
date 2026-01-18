from __future__ import with_statement
from functools import partial
import warnings
from passlib.utils import is_ascii_safe, to_bytes
from passlib.utils.compat import irange, PY2, PY3, u, unicode, join_bytes, PYPY
from passlib.tests.utils import TestCase, hb, run_with_fixed_seeds
from passlib.utils.binary import h64, h64big
def test_is_same_codec(self):
    """test is_same_codec()"""
    from passlib.utils import is_same_codec
    self.assertTrue(is_same_codec(None, None))
    self.assertFalse(is_same_codec(None, 'ascii'))
    self.assertTrue(is_same_codec('ascii', 'ascii'))
    self.assertTrue(is_same_codec('ascii', 'ASCII'))
    self.assertTrue(is_same_codec('utf-8', 'utf-8'))
    self.assertTrue(is_same_codec('utf-8', 'utf8'))
    self.assertTrue(is_same_codec('utf-8', 'UTF_8'))
    self.assertFalse(is_same_codec('ascii', 'utf-8'))