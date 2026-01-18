import io
import os
import sys
import pickle
import subprocess
from test import support
import unittest
from unittest.case import _Outcome
from unittest.test.support import (LoggingResult,
def testBufferAndFailfast(self):

    class Test(unittest.TestCase):

        def testFoo(self):
            pass
    result = unittest.TestResult()
    runner = unittest.TextTestRunner(stream=io.StringIO(), failfast=True, buffer=True)
    runner._makeResult = lambda: result
    runner.run(Test('testFoo'))
    self.assertTrue(result.failfast)
    self.assertTrue(result.buffer)