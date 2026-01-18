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
def test_innerCallbacksPreserved(self) -> None:
    """
        When a L{Deferred} encounters a result which is another L{Deferred}
        which is waiting on a third L{Deferred}, the middle L{Deferred}'s
        callbacks are executed after the third L{Deferred} fires and before the
        first receives a result.
        """
    results: List[Union[Tuple[str, str], str]] = []
    failures: List[Failure] = []
    inner: Deferred[str] = Deferred()

    def cb(result: str) -> Deferred[str]:
        results.append(('start-of-cb', result))
        d = defer.succeed('inner')

        def firstCallback(result: str) -> Deferred[str]:
            results.append(('firstCallback', 'inner'))
            return inner

        def secondCallback(result: str) -> str:
            results.append(('secondCallback', result))
            return result * 2
        d.addCallback(firstCallback).addCallback(secondCallback)
        d.addErrback(failures.append)
        return d
    outer = defer.succeed('outer')
    outer.addCallback(cb)
    inner.callback('orange')
    outer.addCallback(results.append)
    inner.addErrback(failures.append)
    outer.addErrback(failures.append)
    self.assertEqual([], failures)
    self.assertEqual(results, [('start-of-cb', 'outer'), ('firstCallback', 'inner'), ('secondCallback', 'orange'), 'orangeorange'])