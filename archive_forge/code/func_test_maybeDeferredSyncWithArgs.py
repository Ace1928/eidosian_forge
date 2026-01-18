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
def test_maybeDeferredSyncWithArgs(self) -> None:
    """
        L{defer.maybeDeferred} should pass arguments to the called function.
        """

    def plusFive(x: int) -> int:
        return x + 5
    results: List[int] = []
    errors: List[Failure] = []
    d = defer.maybeDeferred(plusFive, 10)
    assert_type(d, Deferred[int])
    d.addCallbacks(results.append, errors.append)
    self.assertEqual(errors, [])
    self.assertEqual(results, [15])