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
def test_cancelBeforeTimeout(self) -> None:
    """
        If the L{Deferred} is manually cancelled before the timeout, it
        is not re-cancelled (no L{AlreadyCancelled} error, and also no
        canceling of inner deferreds), and the default C{onTimeoutCancel}
        function is not called, preserving the original L{CancelledError}.
        """
    clock = Clock()
    d: Deferred[None] = Deferred()
    d.addTimeout(10, clock)
    innerDeferred: Deferred[None] = Deferred()
    dCanceled = None

    def onErrback(f: Failure) -> Deferred[None]:
        nonlocal dCanceled
        dCanceled = f
        return innerDeferred
    d.addErrback(onErrback)
    d.cancel()
    assert dCanceled is not None
    self.assertIsInstance(dCanceled, Failure)
    self.assertIs(dCanceled.type, defer.CancelledError)
    clock.advance(15)
    self.assertNoResult(innerDeferred)