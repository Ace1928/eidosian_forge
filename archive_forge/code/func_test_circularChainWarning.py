from __future__ import annotations
import contextvars
import functools
import gc
import re
import traceback
import types
import unittest as pyunit
import warnings
import weakref
from asyncio import (
from typing import (
from hamcrest import assert_that, empty, equal_to
from hypothesis import given
from hypothesis.strategies import integers
from typing_extensions import assert_type
from twisted.internet import defer, reactor
from twisted.internet.defer import (
from twisted.internet.task import Clock
from twisted.python import log
from twisted.python.compat import _PYPY
from twisted.python.failure import Failure
from twisted.trial import unittest
def test_circularChainWarning(self) -> None:
    """
        When a Deferred is returned from a callback directly attached to that
        same Deferred, a warning is emitted.
        """
    d: Deferred[str] = Deferred()

    def circularCallback(result: str) -> Deferred[str]:
        return d
    d.addCallback(circularCallback)
    d.callback('foo')
    circular_warnings = self.flushWarnings([circularCallback])
    self.assertEqual(len(circular_warnings), 1)
    warning = circular_warnings[0]
    self.assertEqual(warning['category'], DeprecationWarning)
    pattern = 'Callback returned the Deferred it was attached to'
    self.assertTrue(re.search(pattern, warning['message']), '\nExpected match: {!r}\nGot: {!r}'.format(pattern, warning['message']))