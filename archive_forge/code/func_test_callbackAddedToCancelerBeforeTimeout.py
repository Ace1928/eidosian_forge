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
def test_callbackAddedToCancelerBeforeTimeout(self) -> None:
    """
        Given a deferred with a cancellation function that resumes the
        callback chain, a callback that is added to the deferred
        before a timeout is added to runs when the timeout fires.  The
        deferred completes successfully, without a
        L{defer.TimeoutError}.
        """
    clock = Clock()
    success = 'success'
    d: Deferred[str] = Deferred(lambda d: d.callback(success))
    dCallbacked = None

    def callback(value: str) -> str:
        nonlocal dCallbacked
        dCallbacked = value
        return value
    d.addCallback(callback)
    d.addTimeout(10, clock)
    clock.advance(15)
    self.assertEqual(dCallbacked, success)
    self.assertIs(success, self.successResultOf(d))