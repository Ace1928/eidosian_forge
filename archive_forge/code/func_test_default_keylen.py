from __future__ import with_statement, division
from binascii import hexlify
import hashlib
import warnings
from passlib.exc import UnknownHashError
from passlib.utils.compat import PY3, u, JYTHON
from passlib.tests.utils import TestCase, TEST_MODE, skipUnless, hb
from passlib.crypto.digest import pbkdf2_hmac, PBKDF2_BACKENDS
def test_default_keylen(self):
    """test keylen==None"""

    def helper(secret=b'password', salt=b'salt', rounds=1, keylen=None, digest='sha1'):
        return pbkdf2_hmac(digest, secret, salt, rounds, keylen)
    self.assertEqual(len(helper(digest='sha1')), 20)
    self.assertEqual(len(helper(digest='sha256')), 32)