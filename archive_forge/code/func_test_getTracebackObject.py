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
def test_getTracebackObject(self) -> None:
    """
        If the C{Failure} has not been cleaned, then C{getTracebackObject}
        returns the traceback object that captured in its constructor.
        """
    f = getDivisionFailure()
    self.assertEqual(f.getTracebackObject(), f.tb)