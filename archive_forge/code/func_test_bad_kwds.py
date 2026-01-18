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
def test_bad_kwds(self):
    stub = SaltedHash(use_defaults=True)._stub_checksum
    self.assertRaises(TypeError, SaltedHash, checksum=stub, salt=None)
    self.assertRaises(ValueError, SaltedHash, checksum=stub, salt='xxx')