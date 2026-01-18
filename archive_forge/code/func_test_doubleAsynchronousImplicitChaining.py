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
def test_doubleAsynchronousImplicitChaining(self) -> None:
    """
        L{Deferred} chaining is transitive.

        In other words, let A, B, and C be Deferreds.  If C is returned from a
        callback on B and B is returned from a callback on A then when C fires,
        A fires.
        """
    first: Deferred[object] = Deferred()
    second: Deferred[object] = Deferred()
    second.addCallback(lambda ign: first)
    third: Deferred[object] = Deferred()
    third.addCallback(lambda ign: second)
    thirdResult: List[object] = []
    third.addCallback(thirdResult.append)
    result = object()
    second.callback(None)
    third.callback(None)
    self.assertEqual(thirdResult, [])
    first.callback(result)
    self.assertEqual(thirdResult, [result])