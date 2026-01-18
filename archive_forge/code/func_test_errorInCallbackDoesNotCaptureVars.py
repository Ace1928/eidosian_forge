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
def test_errorInCallbackDoesNotCaptureVars(self) -> None:
    """
        An error raised by a callback creates a Failure.  The Failure captures
        locals and globals if and only if C{Deferred.debug} is set.
        """
    d: Deferred[None] = Deferred()
    d.callback(None)
    defer.setDebugging(False)

    def raiseError(ignored: object) -> None:
        raise GenericError('Bang')
    d.addCallback(raiseError)
    l: List[Failure] = []
    d.addErrback(l.append)
    fail = l[0]
    localz, globalz = fail.frames[0][-2:]
    self.assertEqual([], localz)
    self.assertEqual([], globalz)