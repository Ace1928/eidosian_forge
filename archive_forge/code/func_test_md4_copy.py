from __future__ import with_statement, division
from binascii import hexlify
import hashlib
from passlib.utils.compat import bascii_to_str, PY3, u
from passlib.crypto.digest import lookup_hash
from passlib.tests.utils import TestCase, skipUnless
def test_md4_copy(self):
    """copy() method"""
    md4 = self.get_md4_const()
    h = md4(b'abc')
    h2 = h.copy()
    h2.update(b'def')
    self.assertEqual(h2.hexdigest(), '804e7f1c2586e50b49ac65db5b645131')
    h.update(b'ghi')
    self.assertEqual(h.hexdigest(), 'c5225580bfe176f6deeee33dee98732c')