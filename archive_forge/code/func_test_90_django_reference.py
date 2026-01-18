from __future__ import with_statement
import logging; log = logging.getLogger(__name__)
import re
import warnings
from passlib import hash
from passlib.utils import repeat_string
from passlib.utils.compat import u
from passlib.tests.utils import TestCase, HandlerCase, skipUnless, SkipTest
from passlib.tests.test_handlers import UPASS_USD, UPASS_TABLE
from passlib.tests.test_ext_django import DJANGO_VERSION, MIN_DJANGO_VERSION, \
from passlib.tests.test_handlers_argon2 import _base_argon2_test
def test_90_django_reference(self):
    """run known correct hashes through Django's check_password()"""
    self._require_django_support()
    from django.contrib.auth.hashers import check_password
    assert self.known_correct_hashes
    for secret, hash in self.iter_known_hashes():
        self.assertTrue(check_password(secret, hash), 'secret=%r hash=%r failed to verify' % (secret, hash))
        self.assertFalse(check_password('x' + secret, hash), 'mangled secret=%r hash=%r incorrect verified' % (secret, hash))