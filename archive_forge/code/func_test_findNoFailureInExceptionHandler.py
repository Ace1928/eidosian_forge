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
def test_findNoFailureInExceptionHandler(self) -> None:
    """
        Within an exception handler, _findFailure should return
        L{None} in case no Failure is associated with the current
        exception.
        """
    try:
        1 / 0
    except BaseException:
        self.assertIsNone(failure.Failure._findFailure())
    else:
        self.fail('No exception raised from 1/0!?')