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
def test_contextvarsWithAsyncAwait(self) -> None:
    """
        When a coroutine is called, the context is taken from when it was first
        called. When it resumes, the same context is applied.
        """
    clock = Clock()
    var: contextvars.ContextVar[int] = contextvars.ContextVar('testvar')
    var.set(1)
    mutatingDeferred: Deferred[bool] = Deferred()
    mutatingDeferred.addCallback(lambda _: var.set(3))
    mutatingDeferredThatFails: Deferred[bool] = Deferred()
    mutatingDeferredThatFails.addCallback(lambda _: var.set(4))
    mutatingDeferredThatFails.addCallback(lambda _: 1 / 0)

    async def asyncFuncAwaitingDeferred() -> None:
        d: Deferred[bool] = Deferred()
        clock.callLater(1, d.callback, True)
        await d
        var.set(3)

    async def testFunction() -> bool:
        self.assertEqual(var.get(), 2)
        await defer.succeed(1)
        self.assertEqual(var.get(), 2)
        clock.callLater(0, mutatingDeferred.callback, True)
        await mutatingDeferred
        self.assertEqual(var.get(), 2)
        clock.callLater(1, mutatingDeferredThatFails.callback, True)
        try:
            await mutatingDeferredThatFails
        except Exception:
            self.assertEqual(var.get(), 2)
        else:
            raise Exception('???? should have failed')
        await asyncFuncAwaitingDeferred()
        self.assertEqual(var.get(), 3)
        return True
    var.set(2)
    d = ensureDeferred(testFunction())
    clock.advance(1)
    clock.advance(1)
    clock.advance(1)
    self.assertEqual(self.successResultOf(d), True)