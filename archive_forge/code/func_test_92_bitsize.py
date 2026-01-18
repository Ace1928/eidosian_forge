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
def test_92_bitsize(self):
    """test bitsize()"""
    from passlib import hash
    self.assertEqual(hash.des_crypt.bitsize(), {'checksum': 66, 'salt': 12})
    self.assertEqual(hash.bcrypt.bitsize(), {'checksum': 186, 'salt': 132})
    self.patchAttr(hash.sha256_crypt, 'default_rounds', 1 << 14 + 3)
    self.assertEqual(hash.sha256_crypt.bitsize(), {'checksum': 258, 'rounds': 14, 'salt': 96})
    self.patchAttr(hash.pbkdf2_sha1, 'default_rounds', 1 << 13 + 3)
    self.assertEqual(hash.pbkdf2_sha1.bitsize(), {'checksum': 160, 'rounds': 13, 'salt': 128})