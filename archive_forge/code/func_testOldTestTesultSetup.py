import io
import sys
import textwrap
from test.support import warnings_helper, captured_stdout, captured_stderr
import traceback
import unittest
from unittest.util import strclass
def testOldTestTesultSetup(self):

    class Test(unittest.TestCase):

        def setUp(self):
            self.skipTest('no reason')

        def testFoo(self):
            pass
    self.assertOldResultWarning(Test('testFoo'), 0)