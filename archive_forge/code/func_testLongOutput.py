import io
import sys
import textwrap
from test.support import warnings_helper, captured_stdout, captured_stderr
import traceback
import unittest
from unittest.util import strclass
def testLongOutput(self):
    classname = f'{__name__}.{self.Test.__qualname__}'
    self.assertEqual(self._run_test('testSuccess', 2), f'testSuccess ({classname}.testSuccess) ... ok\n')
    self.assertEqual(self._run_test('testSkip', 2), f"testSkip ({classname}.testSkip) ... skipped 'skip'\n")
    self.assertEqual(self._run_test('testFail', 2), f'testFail ({classname}.testFail) ... FAIL\n')
    self.assertEqual(self._run_test('testError', 2), f'testError ({classname}.testError) ... ERROR\n')
    self.assertEqual(self._run_test('testExpectedFailure', 2), f'testExpectedFailure ({classname}.testExpectedFailure) ... expected failure\n')
    self.assertEqual(self._run_test('testUnexpectedSuccess', 2), f'testUnexpectedSuccess ({classname}.testUnexpectedSuccess) ... unexpected success\n')