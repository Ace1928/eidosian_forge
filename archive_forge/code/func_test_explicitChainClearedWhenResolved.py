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
def test_explicitChainClearedWhenResolved(self) -> None:
    """
        Any recorded chaining is cleared once the chaining is resolved, since
        it no longer exists.

        In other words, if one L{Deferred} is recorded as depending on the
        result of another, and I{that} L{Deferred} has fired, then the
        dependency is resolved and we no longer benefit from recording it.
        """
    a: Deferred[None] = Deferred()
    b: Deferred[None] = Deferred()
    b.chainDeferred(a)
    b.callback(None)
    self.assertIsNone(a._chainedTo)