from __future__ import with_statement, division
from binascii import hexlify
import hashlib
import warnings
from passlib.exc import UnknownHashError
from passlib.utils.compat import PY3, u, JYTHON
from passlib.tests.utils import TestCase, TEST_MODE, skipUnless, hb
from passlib.crypto.digest import pbkdf2_hmac, PBKDF2_BACKENDS
def test_lookup_hash_ctor(self):
    """lookup_hash() -- constructor"""
    from passlib.crypto.digest import lookup_hash
    self.assertRaises(ValueError, lookup_hash, 'new')
    self.assertRaises(ValueError, lookup_hash, '__name__')
    self.assertRaises(ValueError, lookup_hash, 'sha4')
    self.assertEqual(lookup_hash('md5'), (hashlib.md5, 16, 64))
    try:
        hashlib.new('sha')
        has_sha = True
    except ValueError:
        has_sha = False
    if has_sha:
        record = lookup_hash('sha')
        const = record[0]
        self.assertEqual(record, (const, 20, 64))
        self.assertEqual(hexlify(const(b'abc').digest()), b'0164b8a914cd2a5e74c4f7ff082c4d97f1edf880')
    else:
        self.assertRaises(ValueError, lookup_hash, 'sha')
    try:
        hashlib.new('md4')
        has_md4 = True
    except ValueError:
        has_md4 = False
    record = lookup_hash('md4')
    const = record[0]
    if not has_md4:
        from passlib.crypto._md4 import md4
        self.assertIs(const, md4)
    self.assertEqual(record, (const, 16, 64))
    self.assertEqual(hexlify(const(b'abc').digest()), b'a448017aaf21d8525fc10ae87aa6729d')
    self.assertIs(lookup_hash('md5'), lookup_hash('md5'))