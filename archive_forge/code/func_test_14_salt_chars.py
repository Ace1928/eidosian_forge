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
def test_14_salt_chars(self):
    """test hash() honors salt_chars"""
    self.require_salt_info()
    handler = self.handler
    mx = handler.max_salt_size
    mn = handler.min_salt_size
    cs = handler.salt_chars
    raw = isinstance(cs, bytes)
    for salt in batch(cs, mx or 32):
        if len(salt) < mn:
            salt = repeat_string(salt, mn)
        salt = self.prepare_salt(salt)
        self.do_stub_encrypt(salt=salt)
    source = u('\x00ÿ')
    if raw:
        source = source.encode('latin-1')
    chunk = max(mn, 1)
    for c in source:
        if c not in cs:
            self.assertRaises(ValueError, self.do_stub_encrypt, salt=c * chunk, __msg__='invalid salt char %r:' % (c,))