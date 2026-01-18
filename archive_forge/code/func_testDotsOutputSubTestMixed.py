import io
import sys
import textwrap
from test.support import warnings_helper, captured_stdout, captured_stderr
import traceback
import unittest
from unittest.util import strclass
def testDotsOutputSubTestMixed(self):
    self.assertEqual(self._run_test('testSubTestMixed', 1), 'sFE')