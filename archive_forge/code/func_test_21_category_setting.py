from __future__ import absolute_import, division, print_function
import logging; log = logging.getLogger(__name__)
import sys
import re
from passlib import apps as _apps, exc, registry
from passlib.apps import django10_context, django14_context, django16_context
from passlib.context import CryptContext
from passlib.ext.django.utils import (
from passlib.utils.compat import iteritems, get_method_function, u
from passlib.utils.decor import memoized_property
from passlib.tests.utils import TestCase, TEST_MODE, handler_derived_from
from passlib.tests.test_handlers import get_handler_case
from passlib.hash import django_pbkdf2_sha256
def test_21_category_setting(self):
    """test PASSLIB_GET_CATEGORY parameter"""
    config = dict(schemes=['sha256_crypt'], sha256_crypt__default_rounds=1000, staff__sha256_crypt__default_rounds=2000, superuser__sha256_crypt__default_rounds=3000)
    from passlib.hash import sha256_crypt

    def run(**kwds):
        """helper to take in user opts, return rounds used in password"""
        user = FakeUser(**kwds)
        user.set_password('stub')
        return sha256_crypt.from_string(user.password).rounds
    self.load_extension(PASSLIB_CONFIG=config)
    self.assertEqual(run(), 1000)
    self.assertEqual(run(is_staff=True), 2000)
    self.assertEqual(run(is_superuser=True), 3000)

    def get_category(user):
        return user.first_name or None
    self.load_extension(PASSLIB_CONTEXT=config, PASSLIB_GET_CATEGORY=get_category)
    self.assertEqual(run(), 1000)
    self.assertEqual(run(first_name='other'), 1000)
    self.assertEqual(run(first_name='staff'), 2000)
    self.assertEqual(run(first_name='superuser'), 3000)

    def get_category(user):
        return None
    self.load_extension(PASSLIB_CONTEXT=config, PASSLIB_GET_CATEGORY=get_category)
    self.assertEqual(run(), 1000)
    self.assertEqual(run(first_name='other'), 1000)
    self.assertEqual(run(first_name='staff', is_staff=True), 1000)
    self.assertEqual(run(first_name='superuser', is_superuser=True), 1000)
    self.assertRaises(TypeError, self.load_extension, PASSLIB_CONTEXT=config, PASSLIB_GET_CATEGORY='x')