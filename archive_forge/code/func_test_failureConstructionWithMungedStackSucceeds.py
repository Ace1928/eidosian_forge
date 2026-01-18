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
@skipIf(raiser is None, 'raiser extension not available')
def test_failureConstructionWithMungedStackSucceeds(self) -> None:
    """
        Pyrex and Cython are known to insert fake stack frames so as to give
        more Python-like tracebacks. These stack frames with empty code objects
        should not break extraction of the exception.
        """
    try:
        raiser.raiseException()
    except raiser.RaiserException:
        f = failure.Failure()
        self.assertTrue(f.check(raiser.RaiserException))
    else:
        self.fail('No exception raised from extension?!')