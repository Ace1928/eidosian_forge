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
def test_20_optional_rounds_attributes(self):
    """validate optional rounds attributes"""
    self.require_rounds_info()
    cls = self.handler
    AssertionError = self.failureException
    if cls.max_rounds is None:
        raise AssertionError('max_rounds not specified')
    if cls.max_rounds < 1:
        raise AssertionError('max_rounds must be >= 1')
    if cls.min_rounds < 0:
        raise AssertionError('min_rounds must be >= 0')
    if cls.min_rounds > cls.max_rounds:
        raise AssertionError('min_rounds must be <= max_rounds')
    if cls.default_rounds is not None:
        if cls.default_rounds < cls.min_rounds:
            raise AssertionError('default_rounds must be >= min_rounds')
        if cls.default_rounds > cls.max_rounds:
            raise AssertionError('default_rounds must be <= max_rounds')
    if cls.rounds_cost not in rounds_cost_values:
        raise AssertionError('unknown rounds cost constant: %r' % (cls.rounds_cost,))