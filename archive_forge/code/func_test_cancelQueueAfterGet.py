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
def test_cancelQueueAfterGet(self) -> None:
    """
        When canceling a L{Deferred} from a L{DeferredQueue} that does not
        have a result (i.e., the L{Deferred} has not fired), the cancel
        causes a L{defer.CancelledError} failure. If the queue has a result
        later on, it doesn't try to fire the deferred.
        """
    queue: DeferredQueue[None] = DeferredQueue()
    d = queue.get()
    d.cancel()
    self.assertImmediateFailure(d, defer.CancelledError)

    def cb(ignore: object) -> Deferred[None]:
        queue.put(None)
        return queue.get().addCallback(self.assertIs, None)
    d.addCallback(cb)
    done: List[None] = []
    d.addCallback(done.append)
    self.assertEqual(len(done), 1)