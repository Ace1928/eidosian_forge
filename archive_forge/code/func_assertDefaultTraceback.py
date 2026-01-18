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
def assertDefaultTraceback(self, captureVars: bool=False) -> None:
    """
        Assert that L{printTraceback} produces and prints a default traceback.

        The default traceback consists of a header::

          Traceback (most recent call last):

        The body with traceback::

          File "/twisted/trial/_synctest.py", line 1180, in _run
             runWithWarningsSuppressed(suppress, method)

        And the footer::

          --- <exception caught here> ---
            File "twisted/test/test_failure.py", line 39, in getDivisionFailure
              1 / 0
            exceptions.ZeroDivisionError: float division

        @param captureVars: Enables L{Failure.captureVars}.
        @type captureVars: C{bool}
        """
    if captureVars:
        exampleLocalVar = 'xyzzy'
        exampleLocalVar
    f = getDivisionFailure(captureVars=captureVars)
    out = StringIO()
    f.printTraceback(out)
    tb = out.getvalue()
    stack = ''
    for method, filename, lineno, localVars, globalVars in f.frames:
        stack += f'  File "{filename}", line {lineno}, in {method}\n'
        stack += f'    {linecache.getline(filename, lineno).strip()}\n'
    self.assertTracebackFormat(tb, 'Traceback (most recent call last):', '%s\n%s%s: %s\n' % (failure.EXCEPTION_CAUGHT_HERE, stack, reflect.qual(f.type), reflect.safe_str(f.value)))
    if captureVars:
        self.assertIsNone(re.search('exampleLocalVar.*xyzzy', tb))