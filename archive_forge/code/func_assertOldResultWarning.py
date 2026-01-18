import io
import sys
import textwrap
from test.support import warnings_helper, captured_stdout, captured_stderr
import traceback
import unittest
from unittest.util import strclass
def assertOldResultWarning(self, test, failures):
    with warnings_helper.check_warnings(('TestResult has no add.+ method,', RuntimeWarning)):
        result = OldResult()
        test.run(result)
        self.assertEqual(len(result.failures), failures)