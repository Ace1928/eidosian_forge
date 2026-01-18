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
def test_coroutineNoCircularReference(self) -> None:
    """
        Tests that there is no circular dependency when using
        L{Deferred.fromCoroutine}, so that the machinery gets cleaned up
        immediately rather than waiting for a GC.
        """
    obj: Set[Any] = set()
    objWeakRef = weakref.ref(obj)

    async def func(a: Any) -> Any:
        return a
    funcD = Deferred.fromCoroutine(func(obj))
    self.assertEqual(obj, self.successResultOf(funcD))
    funcDWeakRef = weakref.ref(funcD)
    del obj
    del funcD
    self.assertIsNone(objWeakRef())
    self.assertIsNone(funcDWeakRef())