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
def assertTracebackFormat(self, tb: str, prefix: str, suffix: str) -> None:
    """
        Assert that the C{tb} traceback contains a particular C{prefix} and
        C{suffix}.

        @param tb: The traceback string.
        @type tb: C{str}
        @param prefix: The string that C{tb} should start with.
        @type prefix: C{str}
        @param suffix: The string that C{tb} should end with.
        @type suffix: C{str}
        """
    self.assertStartsWith(tb, prefix)
    self.assertEndsWith(tb, suffix)