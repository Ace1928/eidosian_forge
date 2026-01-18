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
def test_has_rounds_using_harness(self):
    """
        HasRounds.using() -- sanity check test harness
        """
    self.require_rounds_info()
    handler = self.handler
    orig_min_rounds = handler.min_rounds
    orig_max_rounds = handler.max_rounds
    orig_default_rounds = handler.default_rounds
    handler, subcls, small, medium, large, adj = self._create_using_rounds_helper()
    self.assertEqual(handler.min_rounds, orig_min_rounds)
    self.assertEqual(handler.max_rounds, orig_max_rounds)
    self.assertEqual(handler.min_desired_rounds, None)
    self.assertEqual(handler.max_desired_rounds, None)
    self.assertEqual(handler.default_rounds, orig_default_rounds)
    self.assertEqual(subcls.min_rounds, orig_min_rounds)
    self.assertEqual(subcls.max_rounds, orig_max_rounds)
    self.assertEqual(subcls.default_rounds, medium)
    self.assertEqual(subcls.min_desired_rounds, small)
    self.assertEqual(subcls.max_desired_rounds, large)