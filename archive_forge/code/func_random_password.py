from __future__ import with_statement
import logging; log = logging.getLogger(__name__)
import os
import sys
import warnings
from passlib import exc, hash
from passlib.utils import repeat_string
from passlib.utils.compat import irange, PY3, u, get_method_function
from passlib.tests.utils import TestCase, HandlerCase, skipUnless, \
def random_password(self):
    while True:
        pwd = super(ldap_plaintext_test.FuzzHashGenerator, self).random_password()
        if pwd:
            return pwd