import io
import sys
import textwrap
from test.support import warnings_helper, captured_stdout, captured_stderr
import traceback
import unittest
from unittest.util import strclass
def testDotsOutputTearDownFail(self):
    out = self._run_test('testSuccess', 1, AssertionError('fail'))
    self.assertEqual(out, 'F')
    out = self._run_test('testError', 1, AssertionError('fail'))
    self.assertEqual(out, 'EF')
    out = self._run_test('testFail', 1, Exception('error'))
    self.assertEqual(out, 'FE')
    out = self._run_test('testSkip', 1, AssertionError('fail'))
    self.assertEqual(out, 'sF')