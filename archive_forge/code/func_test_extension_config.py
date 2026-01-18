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
def test_extension_config(self):
    """
        test extension config is loaded correctly
        """
    if not self.patched:
        raise self.skipTest('extension not loaded')
    ctx = self.context
    from django.contrib.auth.hashers import check_password
    from passlib.ext.django.models import password_context
    self.assertEqual(password_context.to_dict(resolve=True), ctx.to_dict(resolve=True))
    from django.contrib.auth.models import check_password as check_password2
    self.assertEqual(check_password2, check_password)