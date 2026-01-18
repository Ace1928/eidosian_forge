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
def test_distb(self) -> None:
    """
        The traceback captured by a L{Failure} is compatible with the stdlib
        L{dis.distb} function as used in post-mortem debuggers. Specifically,
        it doesn't cause that function to raise an exception.
        """
    f = getDivisionFailure()
    buf = StringIO()
    distb(f.getTracebackObject(), file=buf)
    self.assertIn(' --> ', buf.getvalue())