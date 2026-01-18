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
def test_findFailureInGenerator(self) -> None:
    """
        Within an exception handler, it should be possible to find the
        original Failure that caused the current exception (if it was
        caused by throwExceptionIntoGenerator).
        """
    f = getDivisionFailure()
    f.cleanFailure()
    foundFailures = []

    def generator() -> Generator[None, None, None]:
        try:
            yield
        except BaseException:
            foundFailures.append(failure.Failure._findFailure())
        else:
            self.fail('No exception sent to generator')
    g = generator()
    next(g)
    self._throwIntoGenerator(f, g)
    self.assertEqual(foundFailures, [f])