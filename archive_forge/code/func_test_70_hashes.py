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
def test_70_hashes(self):
    """test known hashes"""
    self.assertTrue(self.known_correct_hashes or self.known_correct_configs, "test must set at least one of 'known_correct_hashes' or 'known_correct_configs'")
    saw8bit = False
    for secret, hash in self.iter_known_hashes():
        if self.is_secret_8bit(secret):
            saw8bit = True
        self.assertTrue(self.do_identify(hash), 'identify() failed to identify hash: %r' % (hash,))
        expect_os_crypt_failure = self.expect_os_crypt_failure(secret)
        try:
            self.check_verify(secret, hash, 'verify() of known hash failed: secret=%r, hash=%r' % (secret, hash))
            result = self.do_genhash(secret, hash)
            self.assertIsInstance(result, str, 'genhash() failed to return native string: %r' % (result,))
            if self.handler.is_disabled and self.disabled_contains_salt:
                continue
            self.assertEqual(result, hash, 'genhash() failed to reproduce known hash: secret=%r, hash=%r: result=%r' % (secret, hash, result))
        except MissingBackendError:
            if not expect_os_crypt_failure:
                raise
    if not saw8bit:
        warn('%s: no 8-bit secrets tested' % self.__class__)