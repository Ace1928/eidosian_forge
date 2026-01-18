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
def testEmptyDeferredList(self) -> None:
    result: List[_DeferredListResultListT[None]] = []

    def cb(resultList: _DeferredListResultListT[None], result: List[_DeferredListResultListT[None]]=result) -> None:
        result.append(resultList)
    dl1: Deferred[_DeferredListResultListT[None]] = DeferredList([])
    dl1.addCallbacks(cb)
    self.assertEqual(result, [[]])
    result[:] = []
    dl2: Deferred[_DeferredListSingleResultT[None]] = DeferredList([], fireOnOneCallback=True)
    dl2.addCallbacks(cb)
    self.assertEqual(result, [])