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
def test_63_large_secret(self):
    """test MAX_PASSWORD_SIZE is enforced"""
    from passlib.exc import PasswordSizeError
    from passlib.utils import MAX_PASSWORD_SIZE
    secret = '.' * (1 + MAX_PASSWORD_SIZE)
    hash = self.get_sample_hash()[1]
    err = self.assertRaises(PasswordSizeError, self.do_genhash, secret, hash)
    self.assertEqual(err.max_size, MAX_PASSWORD_SIZE)
    self.assertRaises(PasswordSizeError, self.do_encrypt, secret)
    self.assertRaises(PasswordSizeError, self.do_verify, secret, hash)