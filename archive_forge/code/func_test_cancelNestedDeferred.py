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
def test_cancelNestedDeferred(self) -> None:
    """
        Verify that a Deferred, a, which is waiting on another Deferred, b,
        returned from one of its callbacks, will propagate
        L{defer.CancelledError} when a is cancelled.
        """

    def innerCancel(d: Deferred[object]) -> None:
        self.cancellerCallCount += 1

    def cancel(d: Deferred[object]) -> None:
        self.assertTrue(False)
    b: Deferred[None] = Deferred(canceller=innerCancel)
    a: Deferred[None] = Deferred(canceller=cancel)
    a.callback(None)
    a.addCallback(lambda data: b)
    a.cancel()
    a.addCallbacks(self._callback, self._errback)
    self.assertEqual(self.cancellerCallCount, 1)
    assert self.errbackResults is not None
    self.assertEqual(self.errbackResults.type, defer.CancelledError)