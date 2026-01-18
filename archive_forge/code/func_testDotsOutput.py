import io
import sys
import textwrap
from test.support import warnings_helper, captured_stdout, captured_stderr
import traceback
import unittest
from unittest.util import strclass
def testDotsOutput(self):
    self.assertEqual(self._run_test('testSuccess', 1), '.')
    self.assertEqual(self._run_test('testSkip', 1), 's')
    self.assertEqual(self._run_test('testFail', 1), 'F')
    self.assertEqual(self._run_test('testError', 1), 'E')
    self.assertEqual(self._run_test('testExpectedFailure', 1), 'x')
    self.assertEqual(self._run_test('testUnexpectedSuccess', 1), 'u')