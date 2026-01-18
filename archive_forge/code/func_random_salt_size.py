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
def random_salt_size(self):
    handler = self.handler
    default = handler.default_salt_size
    assert handler.min_salt_size == 0
    lower = 1
    upper = handler.max_salt_size or default * 4
    return self.randintgauss(lower, upper, default, default * 0.5)