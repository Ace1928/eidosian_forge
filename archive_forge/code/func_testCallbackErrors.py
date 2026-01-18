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
def testCallbackErrors(self) -> None:
    l: List[Failure] = []
    d: Deferred[int] = Deferred()
    d.addCallback(lambda _: 1 // 0).addErrback(l.append)
    d.callback(1)
    self.assertIsInstance(l[0].value, ZeroDivisionError)
    l = []
    d2 = Deferred[int]()
    d2.addCallback(lambda _: Failure(ZeroDivisionError())).addErrback(l.append)
    d2.callback(1)
    self.assertIsInstance(l[0].value, ZeroDivisionError)