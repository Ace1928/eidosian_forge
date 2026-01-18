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
def test_chainedPausedDeferredWithResult(self) -> None:
    """
        When a paused Deferred with a result is returned from a callback on
        another Deferred, the other Deferred is chained to the first and waits
        for it to be unpaused.
        """
    expected = object()
    paused: Deferred[object] = Deferred()
    paused.callback(expected)
    paused.pause()
    chained: Deferred[None] = Deferred()
    chained.addCallback(lambda ignored: paused)
    chained.callback(None)
    result: List[object] = []
    chained.addCallback(result.append)
    self.assertEqual(result, [])
    paused.unpause()
    self.assertEqual(result, [expected])