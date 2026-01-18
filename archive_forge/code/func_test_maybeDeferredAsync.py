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
def test_maybeDeferredAsync(self) -> None:
    """
        L{defer.maybeDeferred} should let L{Deferred} instance pass by
        so that original result is the same.
        """
    d1: Deferred[str] = Deferred()
    d2 = defer.maybeDeferred(lambda: d1)
    d1.callback('Success')
    result: List[str] = []
    d2.addCallback(result.append)
    self.assertEqual(result, ['Success'])