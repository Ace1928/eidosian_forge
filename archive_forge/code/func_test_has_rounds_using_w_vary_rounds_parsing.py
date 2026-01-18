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
def test_has_rounds_using_w_vary_rounds_parsing(self):
    """
        HasRounds.using() -- vary_rounds parsing
        """
    handler, subcls, small, medium, large, adj = self._create_using_rounds_helper()

    def parse(value):
        return subcls.using(vary_rounds=value).vary_rounds
    self.assertEqual(parse(0.1), 0.1)
    self.assertEqual(parse('0.1'), 0.1)
    self.assertEqual(parse('10%'), 0.1)
    self.assertEqual(parse(1000), 1000)
    self.assertEqual(parse('1000'), 1000)
    self.assertRaises(ValueError, parse, -0.1)
    self.assertRaises(ValueError, parse, 1.1)