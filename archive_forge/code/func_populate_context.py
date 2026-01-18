from __future__ import with_statement
import logging; log = logging.getLogger(__name__)
import os
import sys
import warnings
from passlib import exc, hash
from passlib.utils import repeat_string
from passlib.utils.compat import irange, PY3, u, get_method_function
from passlib.tests.utils import TestCase, HandlerCase, skipUnless, \
def populate_context(self, secret, kwds):
    """insert username into kwds"""
    if isinstance(secret, tuple):
        secret, user, realm = secret
    else:
        user, realm = ('user', 'realm')
    kwds.setdefault('user', user)
    kwds.setdefault('realm', realm)
    return secret