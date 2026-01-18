from __future__ import with_statement
from binascii import unhexlify
import contextlib
from functools import wraps, partial
import hashlib
import logging; log = logging.getLogger(__name__)
import random
import re
import os
import sys
import tempfile
import threading
import time
from passlib.exc import PasslibHashWarning, PasslibConfigWarning
from passlib.utils.compat import PY3, JYTHON
import warnings
from warnings import warn
from passlib import exc
from passlib.exc import MissingBackendError
import passlib.registry as registry
from passlib.tests.backports import TestCase as _TestCase, skip, skipIf, skipUnless, SkipTest
from passlib.utils import has_rounds_info, has_salt_info, rounds_cost_values, \
from passlib.utils.compat import iteritems, irange, u, unicode, PY2, nullcontext
from passlib.utils.decor import classproperty
import passlib.utils.handlers as uh
def test_61_secret_case_sensitive(self):
    """test password case sensitivity"""
    hash_insensitive = self.secret_case_insensitive is True
    verify_insensitive = self.secret_case_insensitive in [True, 'verify-only']
    lower = 'test'
    upper = 'TEST'
    h1 = self.do_encrypt(lower)
    if verify_insensitive and (not self.handler.is_disabled):
        self.assertTrue(self.do_verify(upper, h1), 'verify() should not be case sensitive')
    else:
        self.assertFalse(self.do_verify(upper, h1), 'verify() should be case sensitive')
    h2 = self.do_encrypt(upper)
    if verify_insensitive and (not self.handler.is_disabled):
        self.assertTrue(self.do_verify(lower, h2), 'verify() should not be case sensitive')
    else:
        self.assertFalse(self.do_verify(lower, h2), 'verify() should be case sensitive')
    h2 = self.do_genhash(upper, h1)
    if hash_insensitive or (self.handler.is_disabled and (not self.disabled_contains_salt)):
        self.assertEqual(h2, h1, 'genhash() should not be case sensitive')
    else:
        self.assertNotEqual(h2, h1, 'genhash() should be case sensitive')