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
def assertWarningList(self, wlist=None, desc=None, msg=None):
    """check that warning list (e.g. from catch_warnings) matches pattern"""
    if desc is None:
        assert wlist is not None
        return self._AssertWarningList(self, desc=wlist, msg=msg)
    assert desc is not None
    if not isinstance(desc, (list, tuple)):
        desc = [desc]
    for idx, entry in enumerate(desc):
        if isinstance(entry, str):
            entry = dict(message_re=entry)
        elif isinstance(entry, type) and issubclass(entry, Warning):
            entry = dict(category=entry)
        elif not isinstance(entry, dict):
            raise TypeError('entry must be str, warning, or dict')
        try:
            data = wlist[idx]
        except IndexError:
            break
        self.assertWarning(data, msg=msg, **entry)
    else:
        if len(wlist) == len(desc):
            return
    std = 'expected %d warnings, found %d: wlist=%s desc=%r' % (len(desc), len(wlist), self._formatWarningList(wlist), desc)
    raise self.failureException(self._formatMessage(msg, std))