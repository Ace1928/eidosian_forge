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
def test_asFutureFailure(self) -> None:
    """
        L{Deferred.asFuture} makes a L{asyncio.Future} fire with an
        exception when the given L{Deferred} does.
        """
    d: Deferred[None] = Deferred()
    theFailure = Failure(ZeroDivisionError())
    loop = self.newLoop()
    future = d.asFuture(loop)
    callAllSoonCalls(loop)
    d.errback(theFailure)
    callAllSoonCalls(loop)
    self.assertRaises(ZeroDivisionError, future.result)