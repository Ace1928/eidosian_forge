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
def test_getTracebackObjectFromCaptureVars(self) -> None:
    """
        C{captureVars=True} has no effect on the result of
        C{getTracebackObject}.
        """
    try:
        1 / 0
    except ZeroDivisionError:
        noVarsFailure = failure.Failure()
        varsFailure = failure.Failure(captureVars=True)
    self.assertEqual(noVarsFailure.getTracebackObject(), varsFailure.tb)