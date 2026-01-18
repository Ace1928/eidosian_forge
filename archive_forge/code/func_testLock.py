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
def testLock(self) -> None:
    lock = DeferredLock()
    lock.acquire().addCallback(self._incr)
    self.assertTrue(lock.locked)
    self.assertEqual(self.counter, 1)
    lock.acquire().addCallback(self._incr)
    self.assertTrue(lock.locked)
    self.assertEqual(self.counter, 1)
    lock.release()
    self.assertTrue(lock.locked)
    self.assertEqual(self.counter, 2)
    lock.release()
    self.assertFalse(lock.locked)
    self.assertEqual(self.counter, 2)
    self.assertRaises(TypeError, lock.run)
    firstUnique = object()
    secondUnique = object()
    controlDeferred: Deferred[object] = Deferred()
    result: Optional[object] = None

    def helper(resultValue: object, returnValue: object=None) -> object:
        nonlocal result
        result = resultValue
        return returnValue
    resultDeferred = lock.run(helper, resultValue=firstUnique, returnValue=controlDeferred)
    self.assertTrue(lock.locked)
    self.assertEqual(result, firstUnique)
    resultDeferred.addCallback(helper)
    lock.acquire().addCallback(self._incr)
    self.assertTrue(lock.locked)
    self.assertEqual(self.counter, 2)
    controlDeferred.callback(secondUnique)
    self.assertEqual(result, secondUnique)
    self.assertTrue(lock.locked)
    self.assertEqual(self.counter, 3)
    d = lock.acquire().addBoth(helper)
    d.cancel()
    self.assertIsInstance(result, Failure)
    self.assertEqual(cast(Failure, result).type, defer.CancelledError)
    lock.release()
    self.assertFalse(lock.locked)

    def returnsInt() -> Deferred[int]:
        return defer.succeed(2)

    async def returnsCoroInt() -> int:
        return 1
    assert_type(lock.run(returnsInt), Deferred[int])
    assert_type(lock.run(returnsCoroInt), Deferred[int])