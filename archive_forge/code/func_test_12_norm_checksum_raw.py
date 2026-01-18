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
def test_12_norm_checksum_raw(self):
    """test GenericHandler + HasRawChecksum mixin"""

    class d1(uh.HasRawChecksum, uh.GenericHandler):
        name = 'd1'
        checksum_size = 4

    def norm_checksum(*a, **k):
        return d1(*a, **k).checksum
    self.assertEqual(norm_checksum(b'1234'), b'1234')
    self.assertRaises(TypeError, norm_checksum, u('xxyx'))
    self.assertEqual(d1()._stub_checksum, b'\x00' * 4)