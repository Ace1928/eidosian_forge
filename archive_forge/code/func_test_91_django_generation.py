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
def test_91_django_generation(self):
    """test against output of Django's make_password()"""
    self._require_django_support()
    from passlib.utils import tick
    from django.contrib.auth.hashers import make_password
    name = self.handler.django_name
    end = tick() + self.max_fuzz_time / 2
    generator = self.FuzzHashGenerator(self, self.getRandom())
    while tick() < end:
        secret, other = generator.random_password_pair()
        if not secret:
            continue
        if isinstance(secret, bytes):
            secret = secret.decode('utf-8')
        hash = make_password(secret, hasher=name)
        self.assertTrue(self.do_identify(hash))
        self.assertTrue(self.do_verify(secret, hash))
        self.assertFalse(self.do_verify(other, hash))