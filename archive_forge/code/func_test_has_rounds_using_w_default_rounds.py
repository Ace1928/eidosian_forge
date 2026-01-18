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
def test_has_rounds_using_w_default_rounds(self):
    """
        HasRounds.using() -- default_rounds
        """
    handler, subcls, small, medium, large, adj = self._create_using_rounds_helper()
    orig_max_rounds = handler.max_rounds
    temp = subcls.using(min_rounds=medium + adj)
    self.assertEqual(temp.default_rounds, medium + adj)
    temp = subcls.using(max_rounds=medium - adj)
    self.assertEqual(temp.default_rounds, medium - adj)
    self.assertRaises(ValueError, subcls.using, default_rounds=small - adj)
    if orig_max_rounds:
        self.assertRaises(ValueError, subcls.using, default_rounds=large + adj)
    self.assertEqual(get_effective_rounds(subcls), medium)
    self.assertEqual(get_effective_rounds(subcls, medium + adj), medium + adj)
    temp = handler.using(default_rounds=str(medium))
    self.assertEqual(temp.default_rounds, medium)
    self.assertRaises(ValueError, handler.using, default_rounds=str(medium) + 'xxx')