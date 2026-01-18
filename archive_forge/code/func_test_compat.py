from __future__ import with_statement
from functools import partial
import warnings
from passlib.utils import is_ascii_safe, to_bytes
from passlib.utils.compat import irange, PY2, PY3, u, unicode, join_bytes, PYPY
from passlib.tests.utils import TestCase, hb, run_with_fixed_seeds
from passlib.utils.binary import h64, h64big
def test_compat(self):
    """test compat's lazymodule"""
    from passlib.utils import compat
    self.assertRegex(repr(compat), "^<module 'passlib.utils.compat' from '.*?'>$")
    dir(compat)
    self.assertTrue('UnicodeIO' in dir(compat))
    self.assertTrue('irange' in dir(compat))