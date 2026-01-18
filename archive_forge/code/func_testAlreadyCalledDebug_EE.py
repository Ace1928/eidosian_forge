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
def testAlreadyCalledDebug_EE(self) -> None:
    d: Deferred[str] = Deferred()
    d.addCallbacks(self._callback, self._errback)
    self._err_1(d)
    try:
        self._err_2(d)
    except defer.AlreadyCalledError as e:
        self._check(e, 'testAlreadyCalledDebug_EE', '_err_1', '_err_2')
    else:
        self.fail('second errback failed to raise AlreadyCalledError')