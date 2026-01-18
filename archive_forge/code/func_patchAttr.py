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
def patchAttr(self, obj, attr, value, require_existing=True, wrap=False):
    """monkeypatch object value, restoring original value on cleanup"""
    try:
        orig = getattr(obj, attr)
    except AttributeError:
        if require_existing:
            raise

        def cleanup():
            try:
                delattr(obj, attr)
            except AttributeError:
                pass
        self.addCleanup(cleanup)
    else:
        self.addCleanup(setattr, obj, attr, orig)
    if wrap:
        value = partial(value, orig)
        wraps(orig)(value)
    setattr(obj, attr, value)