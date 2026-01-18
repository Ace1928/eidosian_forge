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
def test_12_min_salt_size(self):
    """test hash() / genconfig() honors min_salt_size"""
    self.require_salt_info()
    handler = self.handler
    salt_char = handler.salt_chars[0:1]
    min_size = handler.min_salt_size
    s1 = salt_char * min_size
    self.do_genconfig(salt=s1)
    self.do_encrypt('stub', salt_size=min_size)
    if min_size > 0:
        self.assertRaises(ValueError, self.do_genconfig, salt=s1[:-1])
    self.assertRaises(ValueError, self.do_encrypt, 'stub', salt_size=min_size - 1)