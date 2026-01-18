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
def test_maybeDeferredCoroutineFailure(self) -> None:
    """
        When called with a coroutine function L{defer.maybeDeferred} returns a
        L{defer.Deferred} which has a L{Failure} result wrapping the exception
        raised by the coroutine function.
        """

    class SomeException(Exception):
        pass

    async def f() -> None:
        raise SomeException()

    def g() -> Coroutine[None, None, None]:
        return f()
    assert_that(self.failureResultOf(defer.maybeDeferred(g)).type, equal_to(SomeException))