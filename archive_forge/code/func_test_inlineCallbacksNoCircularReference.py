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
def test_inlineCallbacksNoCircularReference(self) -> None:
    """
        When using L{defer.inlineCallbacks}, after the function exits, it will
        not keep references to the function itself or the arguments.

        This ensures that the machinery gets deallocated immediately rather than
        waiting for a GC, on CPython.

        The GC on PyPy works differently (del doesn't immediately deallocate the
        object), so we skip the test.
        """
    obj: Set[Any] = set()
    objWeakRef = weakref.ref(obj)

    @defer.inlineCallbacks
    def func(a: Any) -> Any:
        yield a
        return a
    funcD = func(obj)
    self.assertEqual(obj, self.successResultOf(funcD))
    funcDWeakRef = weakref.ref(funcD)
    del obj
    del funcD
    self.assertIsNone(objWeakRef())
    self.assertIsNone(funcDWeakRef())