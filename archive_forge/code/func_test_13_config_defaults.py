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
def test_13_config_defaults(self):
    """test PASSLIB_CONFIG default behavior"""
    from passlib.ext.django.utils import PASSLIB_DEFAULT
    default = CryptContext.from_string(PASSLIB_DEFAULT)
    self.load_extension()
    self.assert_patched(PASSLIB_DEFAULT)
    self.load_extension(PASSLIB_CONTEXT='passlib-default', check=False)
    self.assert_patched(PASSLIB_DEFAULT)
    self.load_extension(PASSLIB_CONTEXT=PASSLIB_DEFAULT, check=False)
    self.assert_patched(PASSLIB_DEFAULT)