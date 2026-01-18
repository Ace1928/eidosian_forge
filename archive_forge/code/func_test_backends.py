from __future__ import with_statement, division
from binascii import hexlify
import hashlib
import warnings
from passlib.exc import UnknownHashError
from passlib.utils.compat import PY3, u, JYTHON
from passlib.tests.utils import TestCase, TEST_MODE, skipUnless, hb
from passlib.crypto.digest import pbkdf2_hmac, PBKDF2_BACKENDS
def test_backends(self):
    """verify expected backends are present"""
    from passlib.crypto.digest import PBKDF2_BACKENDS
    try:
        import fastpbkdf2
        has_fastpbkdf2 = True
    except ImportError:
        has_fastpbkdf2 = False
    self.assertEqual('fastpbkdf2' in PBKDF2_BACKENDS, has_fastpbkdf2)
    try:
        from hashlib import pbkdf2_hmac
        has_hashlib_ssl = pbkdf2_hmac.__module__ != 'hashlib'
    except ImportError:
        has_hashlib_ssl = False
    self.assertEqual('hashlib-ssl' in PBKDF2_BACKENDS, has_hashlib_ssl)
    from passlib.utils.compat import PY3
    if PY3:
        self.assertIn('builtin-from-bytes', PBKDF2_BACKENDS)
    else:
        self.assertIn('builtin-unpack', PBKDF2_BACKENDS)