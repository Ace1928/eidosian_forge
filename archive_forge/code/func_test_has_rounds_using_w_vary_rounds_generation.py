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
def test_has_rounds_using_w_vary_rounds_generation(self):
    """
        HasRounds.using() -- vary_rounds generation
        """
    handler, subcls, small, medium, large, adj = self._create_using_rounds_helper()

    def get_effective_range(cls):
        seen = set((get_effective_rounds(cls) for _ in irange(1000)))
        return (min(seen), max(seen))

    def assert_rounds_range(vary_rounds, lower, upper):
        temp = subcls.using(vary_rounds=vary_rounds)
        seen_lower, seen_upper = get_effective_range(temp)
        self.assertEqual(seen_lower, lower, 'vary_rounds had wrong lower limit:')
        self.assertEqual(seen_upper, upper, 'vary_rounds had wrong upper limit:')
    assert_rounds_range(0, medium, medium)
    assert_rounds_range('0%', medium, medium)
    assert_rounds_range(adj, medium - adj, medium + adj)
    assert_rounds_range(50, max(small, medium - 50), min(large, medium + 50))
    if handler.rounds_cost == 'log2':
        assert_rounds_range('1%', medium, medium)
        assert_rounds_range('49%', medium, medium)
        assert_rounds_range('50%', medium - adj, medium)
    else:
        lower, upper = get_effective_range(subcls.using(vary_rounds='50%'))
        self.assertGreaterEqual(lower, max(small, medium * 0.5))
        self.assertLessEqual(lower, max(small, medium * 0.8))
        self.assertGreaterEqual(upper, min(large, medium * 1.2))
        self.assertLessEqual(upper, min(large, medium * 1.5))