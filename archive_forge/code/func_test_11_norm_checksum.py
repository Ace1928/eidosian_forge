from __future__ import with_statement
import re
import hashlib
from logging import getLogger
import warnings
from passlib.hash import ldap_md5, sha256_crypt
from passlib.exc import MissingBackendError, PasslibHashWarning
from passlib.utils.compat import str_to_uascii, \
import passlib.utils.handlers as uh
from passlib.tests.utils import HandlerCase, TestCase
from passlib.utils.compat import u
def test_11_norm_checksum(self):
    """test GenericHandler checksum handling"""

    class d1(uh.GenericHandler):
        name = 'd1'
        checksum_size = 4
        checksum_chars = u('xz')

    def norm_checksum(checksum=None, **k):
        return d1(checksum=checksum, **k).checksum
    self.assertRaises(ValueError, norm_checksum, u('xxx'))
    self.assertEqual(norm_checksum(u('xxxx')), u('xxxx'))
    self.assertEqual(norm_checksum(u('xzxz')), u('xzxz'))
    self.assertRaises(ValueError, norm_checksum, u('xxxxx'))
    self.assertRaises(ValueError, norm_checksum, u('xxyx'))
    self.assertRaises(TypeError, norm_checksum, b'xxyx')
    self.assertEqual(d1()._stub_checksum, u('xxxx'))