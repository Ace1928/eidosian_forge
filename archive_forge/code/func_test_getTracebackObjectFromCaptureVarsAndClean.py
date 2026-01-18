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
def test_getTracebackObjectFromCaptureVarsAndClean(self) -> None:
    """
        If the Failure was created with captureVars, then C{getTracebackObject}
        returns an object that looks the same to L{traceback.extract_tb}.
        """
    f = getDivisionFailure(captureVars=True)
    expected = traceback.extract_tb(f.getTracebackObject())
    f.cleanFailure()
    observed = traceback.extract_tb(f.getTracebackObject())
    self.assertEqual(expected, observed)