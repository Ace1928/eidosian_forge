from __future__ import with_statement, division
from binascii import hexlify
import hashlib
import warnings
from passlib.exc import UnknownHashError
from passlib.utils.compat import PY3, u, JYTHON
from passlib.tests.utils import TestCase, TEST_MODE, skipUnless, hb
from passlib.crypto.digest import pbkdf2_hmac, PBKDF2_BACKENDS
def test_lookup_hash_metadata(self):
    """lookup_hash() -- metadata"""
    from passlib.crypto.digest import lookup_hash
    info = lookup_hash('sha256')
    self.assertEqual(info.name, 'sha256')
    self.assertEqual(info.iana_name, 'sha-256')
    self.assertEqual(info.block_size, 64)
    self.assertEqual(info.digest_size, 32)
    self.assertIs(lookup_hash('SHA2-256'), info)
    info = lookup_hash('md5')
    self.assertEqual(info.name, 'md5')
    self.assertEqual(info.iana_name, 'md5')
    self.assertEqual(info.block_size, 64)
    self.assertEqual(info.digest_size, 16)