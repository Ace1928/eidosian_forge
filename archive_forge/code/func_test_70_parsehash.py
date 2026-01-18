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
def test_70_parsehash(self):
    """
        parsehash()
        """
    self.require_parsehash()
    handler = self.handler
    hash = self.do_encrypt('stub')
    result = handler.parsehash(hash)
    self.assertIsInstance(result, dict)
    result2 = handler.parsehash(hash, checksum=False)
    correct2 = result.copy()
    correct2.pop('checksum', None)
    self.assertEqual(result2, correct2)
    result3 = handler.parsehash(hash, sanitize=True)
    correct3 = result.copy()
    if PY2:
        warnings.filterwarnings('ignore', '.*unequal comparison failed to convert.*', category=UnicodeWarning)
    for key in ('salt', 'checksum'):
        if key in result3:
            self.assertNotEqual(result3[key], correct3[key])
            self.assert_is_masked(result3[key])
            correct3[key] = result3[key]
    self.assertEqual(result3, correct3)