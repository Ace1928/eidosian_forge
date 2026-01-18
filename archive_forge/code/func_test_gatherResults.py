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
def test_gatherResults(self) -> None:
    results: List[List[int]] = []
    defer.gatherResults([defer.succeed(1), defer.succeed(2)]).addCallback(results.append)
    self.assertEqual(results, [[1, 2]])
    errors: List[Failure] = []
    dl = [defer.succeed(1), defer.fail(ValueError())]
    defer.gatherResults(dl).addErrback(errors.append)
    self.assertEqual(len(errors), 1)
    self.assertIsInstance(errors[0], Failure)
    dl[1].addErrback(lambda e: 1)