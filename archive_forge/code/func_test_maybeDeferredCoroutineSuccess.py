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
def test_maybeDeferredCoroutineSuccess(self) -> None:
    """
        When called with a coroutine function L{defer.maybeDeferred} returns a
        L{defer.Deferred} which has the same result as the coroutine returned
        by the function.
        """

    async def f() -> int:
        return 7
    coroutineDeferred = defer.maybeDeferred(f)
    assert_type(coroutineDeferred, Deferred[int])
    self.assertEqual(self.successResultOf(coroutineDeferred), 7)