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
def test_cancelDeferredListWithFireOnOneCallbackAndDeferredCallback(self) -> None:
    """
        When cancelling an unfired L{DeferredList} with the flag
        C{fireOnOneCallback} set, if one of the L{Deferred} callbacks
        in its canceller, the L{DeferredList} will callback with the
        result and the index of the L{Deferred} in a C{tuple}.
        """
    deferredOne: Deferred[str] = Deferred(fakeCallbackCanceller)
    deferredTwo: Deferred[str] = Deferred()
    deferredList = DeferredList([deferredOne, deferredTwo], fireOnOneCallback=True)
    deferredList.cancel()
    self.failureResultOf(deferredTwo, defer.CancelledError)
    result = self.successResultOf(deferredList)
    self.assertEqual(result, ('Callback Result', 0))