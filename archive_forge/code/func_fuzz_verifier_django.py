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
def fuzz_verifier_django(self):
    try:
        self._require_django_support()
    except SkipTest:
        return None
    from django.contrib.auth.hashers import check_password

    def verify_django(secret, hash):
        """django/check_password"""
        if self.handler.name == 'django_bcrypt' and hash.startswith('bcrypt$$2y$'):
            hash = hash.replace('$$2y$', '$$2a$')
        if isinstance(secret, bytes):
            secret = secret.decode('utf-8')
        return check_password(secret, hash)
    return verify_django