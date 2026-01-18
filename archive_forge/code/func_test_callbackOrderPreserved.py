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
def test_callbackOrderPreserved(self) -> None:
    """
        A callback added to a L{Deferred} after a previous callback attached
        another L{Deferred} as a result is run after the callbacks of the other
        L{Deferred} are run.
        """
    results: List[Tuple[str, Union[str, List[str], None]]] = []
    failures: List[Failure] = []
    a: Deferred[str] = Deferred()

    def cb(result: str) -> Deferred[None]:
        results.append(('cb', result))
        d: Deferred[None] = Deferred()

        def firstCallback(result: None) -> Deferred[List[str]]:
            results.append(('firstCallback', result))
            return defer.gatherResults([a])

        def secondCallback(result: List[str]) -> None:
            results.append(('secondCallback', result))
        returner = d.addCallback(firstCallback).addCallback(secondCallback).addErrback(failures.append)
        d.callback(None)
        return returner
    outer: Deferred[str] = Deferred()
    outer.addCallback(cb)
    outer.addCallback(lambda x: results.append(('final', None)))
    outer.addErrback(failures.append)
    outer.callback('outer')
    self.assertEqual([('cb', 'outer'), ('firstCallback', None)], results)
    a.callback('withers')
    self.assertEqual([], failures)
    self.assertEqual(results, [('cb', 'outer'), ('firstCallback', None), ('secondCallback', ['withers']), ('final', None)])