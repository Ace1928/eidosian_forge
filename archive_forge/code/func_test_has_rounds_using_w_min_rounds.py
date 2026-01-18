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
def test_has_rounds_using_w_min_rounds(self):
    """
        HasRounds.using() -- min_rounds / min_desired_rounds
        """
    handler, subcls, small, medium, large, adj = self._create_using_rounds_helper()
    orig_min_rounds = handler.min_rounds
    orig_max_rounds = handler.max_rounds
    orig_default_rounds = handler.default_rounds
    if orig_min_rounds > 0:
        self.assertRaises(ValueError, handler.using, min_desired_rounds=orig_min_rounds - adj)
        with self.assertWarningList([PasslibHashWarning]):
            temp = handler.using(min_desired_rounds=orig_min_rounds - adj, relaxed=True)
        self.assertEqual(temp.min_desired_rounds, orig_min_rounds)
    if orig_max_rounds:
        self.assertRaises(ValueError, handler.using, min_desired_rounds=orig_max_rounds + adj)
        with self.assertWarningList([PasslibHashWarning]):
            temp = handler.using(min_desired_rounds=orig_max_rounds + adj, relaxed=True)
        self.assertEqual(temp.min_desired_rounds, orig_max_rounds)
    with self.assertWarningList([]):
        temp = subcls.using(min_desired_rounds=small - adj)
    self.assertEqual(temp.min_desired_rounds, small - adj)
    temp = subcls.using(min_desired_rounds=small + 2 * adj)
    self.assertEqual(temp.min_desired_rounds, small + 2 * adj)
    with self.assertWarningList([]):
        temp = subcls.using(min_desired_rounds=large + adj)
    self.assertEqual(temp.min_desired_rounds, large + adj)
    self.assertEqual(get_effective_rounds(subcls, small + adj), small + adj)
    self.assertEqual(get_effective_rounds(subcls, small), small)
    with self.assertWarningList([]):
        self.assertEqual(get_effective_rounds(subcls, small - adj), small - adj)
    temp = handler.using(min_rounds=small)
    self.assertEqual(temp.min_desired_rounds, small)
    temp = handler.using(min_rounds=str(small))
    self.assertEqual(temp.min_desired_rounds, small)
    self.assertRaises(ValueError, handler.using, min_rounds=str(small) + 'xxx')