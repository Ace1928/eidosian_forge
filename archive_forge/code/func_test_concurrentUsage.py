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
def test_concurrentUsage(self) -> Deferred[None]:
    """
        Test that an appropriate exception is raised when attempting
        to use deferUntilLocked concurrently.
        """
    self.lock.lock()
    self.clock.callLater(1, self.lock.unlock)
    d1 = self.lock.deferUntilLocked()
    d2 = self.lock.deferUntilLocked()
    self.assertFailure(d2, defer.AlreadyTryingToLockError)
    self.clock.advance(1)
    return d1