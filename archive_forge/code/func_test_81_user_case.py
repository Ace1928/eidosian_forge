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
def test_81_user_case(self):
    """test user case sensitivity"""
    lower = self.default_user.lower()
    upper = lower.upper()
    hash = self.do_encrypt('stub', context=dict(user=lower))
    if self.user_case_insensitive:
        self.assertTrue(self.do_verify('stub', hash, user=upper), 'user should not be case sensitive')
    else:
        self.assertFalse(self.do_verify('stub', hash, user=upper), 'user should be case sensitive')