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
def test_cancelDeferredListWithFireOnOneErrbackAllDeferredsCallback(self) -> None:
    """
        When cancelling an unfired L{DeferredList} with the flag
        C{fireOnOneErrback} set, if all the L{Deferred} callbacks
        in its canceller, the L{DeferredList} will callback with a
        C{list} of (success, result) C{tuple}s.
        """
    deferredOne: Deferred[str] = Deferred(fakeCallbackCanceller)
    deferredTwo: Deferred[str] = Deferred(fakeCallbackCanceller)
    deferredList = DeferredList([deferredOne, deferredTwo], fireOnOneErrback=True)
    deferredList.cancel()
    result = self.successResultOf(deferredList)
    self.assertTrue(result[0][0])
    self.assertEqual(result[0][1], 'Callback Result')
    self.assertTrue(result[1][0])
    self.assertEqual(result[1][1], 'Callback Result')