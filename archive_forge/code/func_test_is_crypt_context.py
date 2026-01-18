from __future__ import with_statement
from functools import partial
import warnings
from passlib.utils import is_ascii_safe, to_bytes
from passlib.utils.compat import irange, PY2, PY3, u, unicode, join_bytes, PYPY
from passlib.tests.utils import TestCase, hb, run_with_fixed_seeds
from passlib.utils.binary import h64, h64big
def test_is_crypt_context(self):
    """test is_crypt_context()"""
    from passlib.utils import is_crypt_context
    from passlib.context import CryptContext
    cc = CryptContext(['des_crypt'])
    self.assertTrue(is_crypt_context(cc))
    self.assertFalse(not is_crypt_context(cc))