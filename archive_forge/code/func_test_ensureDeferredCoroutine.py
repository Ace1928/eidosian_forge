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
def test_ensureDeferredCoroutine(self) -> None:
    """
        L{ensureDeferred} will turn a coroutine into a L{Deferred}.
        """

    async def run() -> str:
        d = defer.succeed('foo')
        res = await d
        return res
    r = run()
    self.assertIsInstance(r, types.CoroutineType)
    d = ensureDeferred(r)
    assert_type(d, Deferred[str])
    self.assertIsInstance(d, Deferred)
    res = self.successResultOf(d)
    self.assertEqual(res, 'foo')