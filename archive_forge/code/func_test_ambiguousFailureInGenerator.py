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
def test_ambiguousFailureInGenerator(self) -> None:
    """
        When a generator reraises a different exception,
        L{Failure._findFailure} inside the generator should find the reraised
        exception rather than original one.
        """

    def generator() -> Generator[None, None, None]:
        try:
            try:
                yield
            except BaseException:
                [][1]
        except BaseException:
            self.assertIsInstance(failure.Failure().value, IndexError)
    g = generator()
    next(g)
    f = getDivisionFailure()
    self._throwIntoGenerator(f, g)