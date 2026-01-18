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
def test_brokenValueBriefDetail(self) -> None:
    """
        A L{Failure} might wrap an exception with a C{__str__} method which
        raises an exception.  In this case, calling C{getTraceback} on the
        failure with the C{"brief"} detail does not raise an exception.
        """
    self._brokenValueTest('brief')