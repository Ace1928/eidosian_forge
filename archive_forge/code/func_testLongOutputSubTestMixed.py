import io
import sys
import textwrap
from test.support import warnings_helper, captured_stdout, captured_stderr
import traceback
import unittest
from unittest.util import strclass
def testLongOutputSubTestMixed(self):
    classname = f'{__name__}.{self.Test.__qualname__}'
    self.assertEqual(self._run_test('testSubTestMixed', 2), f"testSubTestMixed ({classname}.testSubTestMixed) ... \n  testSubTestMixed ({classname}.testSubTestMixed) [skip] (b=2) ... skipped 'skip'\n  testSubTestMixed ({classname}.testSubTestMixed) [fail] (c=3) ... FAIL\n  testSubTestMixed ({classname}.testSubTestMixed) [error] (d=4) ... ERROR\n")