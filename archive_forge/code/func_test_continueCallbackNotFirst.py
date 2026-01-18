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
def test_continueCallbackNotFirst(self) -> None:
    """
        The continue callback of a L{Deferred} waiting for another L{Deferred}
        is not necessarily the first one. This is somewhat a whitebox test
        checking that we search for that callback among the whole list of
        callbacks.
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
    defer.succeed('outer').addCallback(cb).addErrback(failures.append)
    self.assertEqual([('cb', 'outer'), ('firstCallback', None)], results)
    a.callback('withers')
    self.assertEqual([], failures)
    self.assertEqual(results, [('cb', 'outer'), ('firstCallback', None), ('secondCallback', ['withers'])])