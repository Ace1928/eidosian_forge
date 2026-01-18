import io
import sys
import textwrap
from test.support import warnings_helper, captured_stdout, captured_stderr
import traceback
import unittest
from unittest.util import strclass
def testSubTestSuccess(self):
    with self.subTest('one', a=1):
        pass
    with self.subTest('two', b=2):
        pass