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
@pyunit.skipIf(_PYPY, 'GC works differently on PyPy.')
def test_canceller_circular_reference_errback(self) -> None:
    """
        A circular reference between a `Deferred` and its canceller
        is broken when the deferred fails.
        """
    canceller = DummyCanceller()
    weakCanceller = weakref.ref(canceller)
    deferred: Deferred[Any] = Deferred(canceller)
    canceller.deferred = deferred
    weakDeferred = weakref.ref(deferred)
    failure = Failure(Exception('The test demands failures.'))
    deferred.errback(failure)
    self.failureResultOf(deferred)
    del deferred
    del canceller
    self.assertIsNone(weakCanceller())
    self.assertIsNone(weakDeferred())