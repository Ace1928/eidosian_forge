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
def test_00_patch_control(self):
    """test set_django_password_context patch/unpatch"""
    self.load_extension(PASSLIB_CONFIG='disabled', check=False)
    self.assert_unpatched()
    with self.assertWarningList('PASSLIB_CONFIG=None is deprecated'):
        self.load_extension(PASSLIB_CONFIG=None, check=False)
    self.assert_unpatched()
    self.load_extension(PASSLIB_CONFIG='django-1.0', check=False)
    self.assert_patched(context=django10_context)
    self.unload_extension()
    self.load_extension(PASSLIB_CONFIG='django-1.4', check=False)
    self.assert_patched(context=django14_context)
    self.unload_extension()