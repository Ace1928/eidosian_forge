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
def test_cancelDeferredListWithOriginalDeferreds(self) -> None:
    """
        Cancelling a L{DeferredList} will cancel the original
        L{Deferred}s passed in.
        """
    deferredOne: Deferred[None] = Deferred()
    deferredTwo: Deferred[None] = Deferred()
    argumentList = [deferredOne, deferredTwo]
    deferredList = DeferredList(argumentList)
    deferredThree: Deferred[None] = Deferred()
    argumentList.append(deferredThree)
    deferredList.cancel()
    self.failureResultOf(deferredOne, defer.CancelledError)
    self.failureResultOf(deferredTwo, defer.CancelledError)
    self.assertNoResult(deferredThree)