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
def assertBriefTraceback(self, captureVars: bool=False) -> None:
    """
        Assert that L{printBriefTraceback} produces and prints a brief
        traceback.

        The brief traceback consists of a header::

          Traceback: <type 'exceptions.ZeroDivisionError'>: float division

        The body with the stacktrace::

          /twisted/trial/_synctest.py:1180:_run
          /twisted/python/util.py:1076:runWithWarningsSuppressed

        And the footer::

          --- <exception caught here> ---
          /twisted/test/test_failure.py:39:getDivisionFailure

        @param captureVars: Enables L{Failure.captureVars}.
        @type captureVars: C{bool}
        """
    if captureVars:
        exampleLocalVar = 'abcde'
        exampleLocalVar
    f = getDivisionFailure()
    out = StringIO()
    f.printBriefTraceback(out)
    tb = out.getvalue()
    stack = ''
    for method, filename, lineno, localVars, globalVars in f.frames:
        stack += f'{filename}:{lineno}:{method}\n'
    zde = repr(ZeroDivisionError)
    self.assertTracebackFormat(tb, f'Traceback: {zde}: ', f'{failure.EXCEPTION_CAUGHT_HERE}\n{stack}')
    if captureVars:
        self.assertIsNone(re.search('exampleLocalVar.*abcde', tb))