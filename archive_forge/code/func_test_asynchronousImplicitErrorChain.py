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
def test_asynchronousImplicitErrorChain(self) -> None:
    """
        Let C{a} and C{b} be two L{Deferred}s.

        If C{a} has no result and is returned from a callback on C{b} then when
        C{a} fails, C{b}'s result becomes the L{Failure} that was C{a}'s result,
        the result of C{a} becomes L{None} so that no unhandled error is logged
        when it is garbage collected.
        """
    first: Deferred[None] = Deferred()
    second: Deferred[None] = Deferred()
    second.addCallback(lambda ign: first)
    second.callback(None)
    secondError: List[Failure] = []
    second.addErrback(secondError.append)
    firstResult: List[None] = []
    first.addCallback(firstResult.append)
    secondResult: List[None] = []
    second.addCallback(secondResult.append)
    self.assertEqual(firstResult, [])
    self.assertEqual(secondResult, [])
    first.errback(RuntimeError("First Deferred's Failure"))
    self.assertTrue(secondError[0].check(RuntimeError))
    self.assertEqual(firstResult, [None])
    self.assertEqual(len(secondResult), 1)