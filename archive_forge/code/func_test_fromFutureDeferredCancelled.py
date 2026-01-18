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
def test_fromFutureDeferredCancelled(self) -> None:
    """
        L{Deferred.fromFuture} makes a L{Deferred} which, when
        cancelled, cancels the L{asyncio.Future} it was created from.
        """
    loop = self.newLoop()
    cancelled: Future[None] = Future(loop=loop)
    d = Deferred.fromFuture(cancelled)
    d.cancel()
    callAllSoonCalls(loop)
    self.assertEqual(cancelled.cancelled(), True)
    self.assertRaises(CancelledError, cancelled.result)
    self.failureResultOf(d).trap(CancelledError)