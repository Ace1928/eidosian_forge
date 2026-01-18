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
def test_03_hash_workflow(self, use_16_legacy=False):
    """test basic hash-string workflow.

        this tests that hash()'s hashes are accepted
        by verify() and identify(), and regenerated correctly by genhash().
        the test is run against a couple of different stock passwords.
        """
    wrong_secret = 'stub'
    for secret in self.stock_passwords:
        result = self.do_encrypt(secret, use_encrypt=use_16_legacy)
        self.check_returned_native_str(result, 'hash')
        self.check_verify(secret, result)
        self.check_verify(wrong_secret, result, negate=True)
        other = self.do_genhash(secret, result)
        self.check_returned_native_str(other, 'genhash')
        if self.handler.is_disabled and self.disabled_contains_salt:
            self.assertNotEqual(other, result, 'genhash() failed to salt result hash: secret=%r hash=%r: result=%r' % (secret, result, other))
        else:
            self.assertEqual(other, result, 'genhash() failed to reproduce hash: secret=%r hash=%r: result=%r' % (secret, result, other))
        other = self.do_genhash(wrong_secret, result)
        self.check_returned_native_str(other, 'genhash')
        if self.handler.is_disabled and (not self.disabled_contains_salt):
            self.assertEqual(other, result, 'genhash() failed to reproduce disabled-hash: secret=%r hash=%r other_secret=%r: result=%r' % (secret, result, wrong_secret, other))
        else:
            self.assertNotEqual(other, result, 'genhash() duplicated hash: secret=%r hash=%r wrong_secret=%r: result=%r' % (secret, result, wrong_secret, other))
        self.assertTrue(self.do_identify(result))