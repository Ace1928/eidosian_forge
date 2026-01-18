import io
import sys
import textwrap
from test.support import warnings_helper, captured_stdout, captured_stderr
import traceback
import unittest
from unittest.util import strclass
def testFailFast(self):
    result = unittest.TestResult()
    result._exc_info_to_string = lambda *_: ''
    result.failfast = True
    result.addError(None, None)
    self.assertTrue(result.shouldStop)
    result = unittest.TestResult()
    result._exc_info_to_string = lambda *_: ''
    result.failfast = True
    result.addFailure(None, None)
    self.assertTrue(result.shouldStop)
    result = unittest.TestResult()
    result._exc_info_to_string = lambda *_: ''
    result.failfast = True
    result.addUnexpectedSuccess(None)
    self.assertTrue(result.shouldStop)