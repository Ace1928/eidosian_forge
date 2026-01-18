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
def test_chainDeferredRecordsImplicitChain(self) -> None:
    """
        We can chain L{Deferred}s implicitly by adding callbacks that return
        L{Deferred}s. When this chaining happens, we record it explicitly as
        soon as we can find out about it.
        """
    a: Deferred[None] = Deferred()
    b: Deferred[None] = Deferred()
    a.addCallback(lambda ignored: b)
    a.callback(None)
    self.assertIs(a._chainedTo, b)