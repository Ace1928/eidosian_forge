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
def test_failureValueFromFailure(self) -> None:
    """
        A L{failure.Failure} constructed from another
        L{failure.Failure} instance, has its C{value} property set to
        the value of that L{failure.Failure} instance.
        """
    exception = ValueError()
    f1 = failure.Failure(exception)
    f2 = failure.Failure(f1)
    self.assertIs(f2.value, exception)