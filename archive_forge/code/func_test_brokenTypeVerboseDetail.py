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
def test_brokenTypeVerboseDetail(self) -> None:
    """
        Like test_brokenTypeBriefDetail, but for the C{"verbose"} detail case.
        """
    self._brokenTypeTest('verbose')