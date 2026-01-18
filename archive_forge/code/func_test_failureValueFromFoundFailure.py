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
def test_failureValueFromFoundFailure(self) -> None:
    """
        A L{failure.Failure} constructed without a C{exc_value}
        argument, will search for an "original" C{Failure}, and if
        found, its value will be used as the value for the new
        C{Failure}.
        """
    exception = ValueError()
    f1 = failure.Failure(exception)
    try:
        f1.trap(OverflowError)
    except BaseException:
        f2 = failure.Failure()
    self.assertIs(f2.value, exception)