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
def test_01_overwrite_detection(self):
    """test detection of foreign monkeypatching"""
    config = '[passlib]\nschemes=des_crypt\n'
    self.load_extension(PASSLIB_CONFIG=config)
    import django.contrib.auth.models as models
    from passlib.ext.django.models import adapter

    def dummy():
        pass
    orig = models.User.set_password
    models.User.set_password = dummy
    with self.assertWarningList('another library has patched.*User\\.set_password'):
        adapter._manager.check_all()
    models.User.set_password = orig
    orig = models.check_password
    models.check_password = dummy
    with self.assertWarningList('another library has patched.*models:check_password'):
        adapter._manager.check_all()
    models.check_password = orig