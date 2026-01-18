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
def test_findFailure(self) -> None:
    """
        Within an exception handler, it should be possible to find the
        original Failure that caused the current exception (if it was
        caused by raiseException).
        """
    f = getDivisionFailure()
    f.cleanFailure()
    try:
        f.raiseException()
    except BaseException:
        self.assertEqual(failure.Failure._findFailure(), f)
    else:
        self.fail('No exception raised from raiseException!?')