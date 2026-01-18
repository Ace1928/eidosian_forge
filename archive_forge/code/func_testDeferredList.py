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
def testDeferredList(self) -> None:
    ResultList = List[Tuple[bool, Union[str, Failure]]]
    defr1: Deferred[str] = Deferred()
    defr2: Deferred[str] = Deferred()
    defr3: Deferred[str] = Deferred()
    dl = DeferredList([defr1, defr2, defr3])
    result: ResultList = []

    def cb(resultList: ResultList, result: ResultList=result) -> None:
        result.extend(resultList)

    def catch(err: Failure) -> None:
        return None
    dl.addCallbacks(cb, cb)
    defr1.callback('1')
    defr2.addErrback(catch)
    defr2.errback(GenericError('2'))
    defr3.callback('3')
    self.assertEqual([result[0], (result[1][0], str(cast(Failure, result[1][1]).value)), result[2]], [(defer.SUCCESS, '1'), (defer.FAILURE, '2'), (defer.SUCCESS, '3')])