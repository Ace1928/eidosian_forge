from __future__ import annotations
import linecache
import pdb
import re
import sys
import traceback
from dis import distb
from io import StringIO
from traceback import FrameSummary
from types import TracebackType
from typing import Any, Generator
from unittest import skipIf
from cython_test_exception_raiser import raiser
from twisted.python import failure, reflect
from twisted.trial.unittest import SynchronousTestCase
def test_ambiguousFailureFromGenerator(self) -> None:
    """
        When a generator reraises a different exception,
        L{Failure._findFailure} above the generator should find the reraised
        exception rather than original one.
        """

    def generator() -> Generator[None, None, None]:
        try:
            yield
        except BaseException:
            [][1]
    g = generator()
    next(g)
    f = getDivisionFailure()
    try:
        self._throwIntoGenerator(f, g)
    except BaseException:
        self.assertIsInstance(failure.Failure().value, IndexError)