from __future__ import with_statement, division
from binascii import hexlify
import hashlib
from passlib.utils.compat import bascii_to_str, PY3, u
from passlib.crypto.digest import lookup_hash
from passlib.tests.utils import TestCase, skipUnless
def test_md4_update(self):
    """update() method"""
    md4 = self.get_md4_const()
    h = md4(b'')
    self.assertEqual(h.hexdigest(), '31d6cfe0d16ae931b73c59d7e0c089c0')
    h.update(b'a')
    self.assertEqual(h.hexdigest(), 'bde52cb31de33e46245e05fbdbd6fb24')
    h.update(b'bcdefghijklmnopqrstuvwxyz')
    self.assertEqual(h.hexdigest(), 'd79e1c308aa5bbcdeea8ed63df412da9')
    if PY3:
        h = md4()
        self.assertRaises(TypeError, h.update, u('a'))
        self.assertEqual(h.hexdigest(), '31d6cfe0d16ae931b73c59d7e0c089c0')
    else:
        h = md4()
        h.update(u('a'))
        self.assertEqual(h.hexdigest(), 'bde52cb31de33e46245e05fbdbd6fb24')