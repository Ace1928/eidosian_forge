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
def test_cancelDeferUntilLockedWithTimeout(self) -> None:
    """
        When cancel a L{Deferred} returned by
        L{DeferredFilesystemLock.deferUntilLocked}, if the timeout is
        set, the timeout call will be cancelled.
        """
    self.lock.lock()
    deferred = self.lock.deferUntilLocked(timeout=1)
    timeoutCall = self.lock._timeoutCall
    assert timeoutCall is not None
    deferred.cancel()
    self.assertFalse(timeoutCall.active())
    self.assertIsNone(self.lock._timeoutCall)
    self.failureResultOf(deferred, defer.CancelledError)