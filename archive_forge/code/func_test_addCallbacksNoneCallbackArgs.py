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
def test_addCallbacksNoneCallbackArgs(self) -> None:
    """
        If given None as a callback args and kwargs, () and {} are used.
        """
    deferred: Deferred[str] = Deferred()
    deferred.addCallbacks(self._callback, self._errback, cast(Tuple[object], None), cast(Mapping[str, object], None), (), {})
    deferred.callback('hello')
    self.assertIsNone(self.errbackResults)
    self.assertEqual(self.callbackResults, (('hello',), {}))