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
def test_ConstructionFails(self) -> None:
    """
        Creating a Failure with no arguments causes it to try to discover the
        current interpreter exception state.  If no such state exists, creating
        the Failure should raise a synchronous exception.
        """
    self.assertRaises(failure.NoCurrentExceptionError, failure.Failure)