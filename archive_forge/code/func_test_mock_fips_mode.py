from __future__ import with_statement, division
from binascii import hexlify
import hashlib
import warnings
from passlib.exc import UnknownHashError
from passlib.utils.compat import PY3, u, JYTHON
from passlib.tests.utils import TestCase, TEST_MODE, skipUnless, hb
from passlib.crypto.digest import pbkdf2_hmac, PBKDF2_BACKENDS
def test_mock_fips_mode(self):
    """
        lookup_hash() -- test set_mock_fips_mode()
        """
    from passlib.crypto.digest import lookup_hash, _set_mock_fips_mode
    if not lookup_hash('md5', required=False).supported:
        raise self.skipTest('md5 not supported')
    _set_mock_fips_mode()
    self.addCleanup(_set_mock_fips_mode, False)
    pat = "'md5' hash disabled for fips"
    self.assertRaisesRegex(UnknownHashError, pat, lookup_hash, 'md5')
    info = lookup_hash('md5', required=False)
    self.assertRegex(info.error_text, pat)
    self.assertRaisesRegex(UnknownHashError, pat, info.const)
    self.assertEqual(info.digest_size, 16)
    self.assertEqual(info.block_size, 64)