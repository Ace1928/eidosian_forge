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
def test_cleanFailureRemovesTracebackInPython3(self) -> None:
    """
        L{failure.Failure.cleanFailure} sets the C{__traceback__} attribute of
        the exception to L{None} in Python 3.
        """
    f = getDivisionFailure()
    self.assertIsNotNone(f.tb)
    self.assertIs(f.value.__traceback__, f.tb)
    f.cleanFailure()
    self.assertIsNone(f.value.__traceback__)