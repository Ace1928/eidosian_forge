import io
import sys
import textwrap
from test.support import warnings_helper, captured_stdout, captured_stderr
import traceback
import unittest
from unittest.util import strclass
def testLongOutputTearDownFail(self):
    classname = f'{__name__}.{self.Test.__qualname__}'
    out = self._run_test('testSuccess', 2, AssertionError('fail'))
    self.assertEqual(out, f'testSuccess ({classname}.testSuccess) ... FAIL\n')
    out = self._run_test('testError', 2, AssertionError('fail'))
    self.assertEqual(out, f'testError ({classname}.testError) ... ERROR\ntestError ({classname}.testError) ... FAIL\n')
    out = self._run_test('testFail', 2, Exception('error'))
    self.assertEqual(out, f'testFail ({classname}.testFail) ... FAIL\ntestFail ({classname}.testFail) ... ERROR\n')
    out = self._run_test('testSkip', 2, AssertionError('fail'))
    self.assertEqual(out, f"testSkip ({classname}.testSkip) ... skipped 'skip'\ntestSkip ({classname}.testSkip) ... FAIL\n")