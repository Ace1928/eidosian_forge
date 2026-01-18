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
def test_21b_max_rounds(self):
    """test hash() / genconfig() honors max_rounds"""
    self.require_rounds_info()
    handler = self.handler
    max_rounds = handler.max_rounds
    if max_rounds is not None:
        self.assertRaises(ValueError, self.do_genconfig, rounds=max_rounds + 1)
        self.assertRaises(ValueError, self.do_encrypt, 'stub', rounds=max_rounds + 1)
    if max_rounds is None:
        self.do_stub_encrypt(rounds=(1 << 31) - 1)
    else:
        self.do_stub_encrypt(rounds=max_rounds)