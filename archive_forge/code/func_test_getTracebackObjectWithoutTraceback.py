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
def test_getTracebackObjectWithoutTraceback(self) -> None:
    """
        L{failure.Failure}s need not be constructed with traceback objects. If
        a C{Failure} has no traceback information at all, C{getTracebackObject}
        just returns None.

        None is a good value, because traceback.extract_tb(None) -> [].
        """
    f = failure.Failure(Exception('some error'))
    self.assertIsNone(f.getTracebackObject())