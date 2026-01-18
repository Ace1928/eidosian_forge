import io
import sys
import textwrap
from test.support import warnings_helper, captured_stdout, captured_stderr
import traceback
import unittest
from unittest.util import strclass
def getStartedResult(self):
    result = unittest.TestResult()
    result.buffer = True
    result.startTest(self)
    return result