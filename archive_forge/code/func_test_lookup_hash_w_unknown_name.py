from __future__ import with_statement, division
from binascii import hexlify
import hashlib
import warnings
from passlib.exc import UnknownHashError
from passlib.utils.compat import PY3, u, JYTHON
from passlib.tests.utils import TestCase, TEST_MODE, skipUnless, hb
from passlib.crypto.digest import pbkdf2_hmac, PBKDF2_BACKENDS
def test_lookup_hash_w_unknown_name(self):
    """lookup_hash() -- unknown hash name"""
    from passlib.crypto.digest import lookup_hash
    self.assertRaises(UnknownHashError, lookup_hash, 'xxx256')
    info = lookup_hash('xxx256', required=False)
    self.assertFalse(info.supported)
    self.assertRaisesRegex(UnknownHashError, "unknown hash: 'xxx256'", info.const)
    self.assertEqual(info.name, 'xxx256')
    self.assertEqual(info.digest_size, None)
    self.assertEqual(info.block_size, None)
    info2 = lookup_hash('xxx256', required=False)
    self.assertIs(info2, info)