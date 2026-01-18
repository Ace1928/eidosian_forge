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
def test_empty_hash_value(self):
    """
        test how methods handle empty string as hash value
        """
    from django.contrib.auth.hashers import check_password, make_password, is_password_usable, identify_hasher
    user = FakeUser()
    user.password = ''
    self.assertFalse(user.check_password(PASS1))
    self.assertEqual(user.password, '')
    self.assertEqual(user.pop_saved_passwords(), [])
    self.assertEqual(user.has_usable_password(), quirks.empty_is_usable_password)
    self.assertFalse(check_password(PASS1, ''))
    self.assertRaises(ValueError, identify_hasher, '')