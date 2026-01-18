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
def test_chainedErrorCleanup(self) -> None:
    """
        If one Deferred with an error result is returned from a callback on
        another Deferred, when the first Deferred is garbage collected it does
        not log its error.
        """
    d: Deferred[None] = Deferred()
    d.addCallback(lambda ign: defer.fail(RuntimeError('zoop')))
    d.callback(None)
    results: List[None] = []
    errors: List[Failure] = []
    d.addCallbacks(results.append, errors.append)
    self.assertEqual(results, [])
    self.assertEqual(len(errors), 1)
    errors[0].trap(Exception)
    del results, errors, d
    gc.collect()
    self.assertEqual(self._loggedErrors(), [])