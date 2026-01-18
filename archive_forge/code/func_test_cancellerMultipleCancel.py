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
def test_cancellerMultipleCancel(self) -> None:
    """
        Verify that calling cancel multiple times on a deferred with a
        canceller that does not errback results in a
        L{defer.CancelledError} and that subsequent calls to cancel do not
        cause an error and that after all that, the canceller was only
        called once.
        """

    def cancel(d: Deferred[object]) -> None:
        self.cancellerCallCount += 1
    d: Deferred[None] = Deferred(canceller=cancel)
    d.addCallbacks(self._callback, self._errback)
    d.cancel()
    assert self.errbackResults is not None
    self.assertEqual(self.errbackResults.type, defer.CancelledError)
    currentFailure = self.errbackResults
    d.cancel()
    self.assertIs(currentFailure, self.errbackResults)
    self.assertEqual(self.cancellerCallCount, 1)