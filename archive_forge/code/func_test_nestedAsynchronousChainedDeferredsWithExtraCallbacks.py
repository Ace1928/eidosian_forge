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
def test_nestedAsynchronousChainedDeferredsWithExtraCallbacks(self) -> None:
    """
        L{Deferred}s can have callbacks that themselves return L{Deferred}s.
        These L{Deferred}s can have other callbacks added before they are
        returned, which subtly changes the callback chain. When these "inner"
        L{Deferred}s fire (even asynchronously), the outer callback chain
        continues.
        """
    results: List[Any] = []
    failures: List[Failure] = []
    inner: Deferred[str] = Deferred()

    def cb(result: str) -> Deferred[Optional[List[str]]]:
        results.append(('start-of-cb', result))
        d = defer.succeed('inner')

        def firstCallback(result: str) -> Deferred[List[str]]:
            results.append(('firstCallback', result))

            def transform(result: str) -> List[str]:
                return [result]
            return inner.addCallback(transform)

        def secondCallback(result: List[str]) -> List[str]:
            results.append(('secondCallback', result))
            return result * 2
        return d.addCallback(firstCallback).addCallback(secondCallback).addErrback(failures.append)
    outer = defer.succeed('outer')
    outer.addCallback(cb)
    outer.addCallback(results.append)
    self.assertEqual(results, [('start-of-cb', 'outer'), ('firstCallback', 'inner')])
    inner.callback('withers')
    outer.addErrback(failures.append)
    inner.addErrback(failures.append)
    self.assertEqual([], failures, "Got errbacks but wasn't expecting any.")
    self.assertEqual(results, [('start-of-cb', 'outer'), ('firstCallback', 'inner'), ('secondCallback', ['withers']), ['withers', 'withers']])