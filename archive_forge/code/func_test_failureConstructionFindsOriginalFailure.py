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
def test_failureConstructionFindsOriginalFailure(self) -> None:
    """
        When a Failure is constructed in the context of an exception
        handler that is handling an exception raised by
        throwExceptionIntoGenerator, the new Failure should be chained to that
        original Failure.
        """
    f = getDivisionFailure()
    f.cleanFailure()
    original_failure_str = f.getTraceback()
    newFailures = []

    def generator() -> Generator[None, None, None]:
        try:
            yield
        except BaseException:
            newFailures.append(failure.Failure())
        else:
            self.fail('No exception sent to generator')
    g = generator()
    next(g)
    self._throwIntoGenerator(f, g)
    self.assertEqual(len(newFailures), 1)
    self.assertEqual(original_failure_str, f.getTraceback())
    self.assertNotEqual(newFailures[0].getTraceback(), f.getTraceback())
    self.assertIn('generator', newFailures[0].getTraceback())
    self.assertNotIn('generator', f.getTraceback())