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
def test_errorLogWithInnerFrameRef(self) -> None:
    """
        Same as L{test_errorLog}, but with an inner frame.
        """

    def _subErrorLogWithInnerFrameRef() -> None:
        d: Deferred[int] = Deferred()
        d.addCallback(lambda x: 1 // 0)
        d.callback(1)
    _subErrorLogWithInnerFrameRef()
    gc.collect()
    self._check()