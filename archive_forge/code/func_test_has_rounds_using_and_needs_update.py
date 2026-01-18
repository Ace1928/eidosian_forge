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
def test_has_rounds_using_and_needs_update(self):
    """
        HasRounds.using() -- desired_rounds + needs_update()
        """
    handler, subcls, small, medium, large, adj = self._create_using_rounds_helper()
    temp = subcls.using(min_desired_rounds=small + 2, max_desired_rounds=large - 2)
    small_hash = self.do_stub_encrypt(subcls, rounds=small)
    medium_hash = self.do_stub_encrypt(subcls, rounds=medium)
    large_hash = self.do_stub_encrypt(subcls, rounds=large)
    self.assertFalse(subcls.needs_update(small_hash))
    self.assertFalse(subcls.needs_update(medium_hash))
    self.assertFalse(subcls.needs_update(large_hash))
    self.assertTrue(temp.needs_update(small_hash))
    self.assertFalse(temp.needs_update(medium_hash))
    self.assertTrue(temp.needs_update(large_hash))