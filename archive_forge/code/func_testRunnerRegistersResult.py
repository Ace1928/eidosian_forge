import io
import os
import sys
import pickle
import subprocess
from test import support
import unittest
from unittest.case import _Outcome
from unittest.test.support import (LoggingResult,
def testRunnerRegistersResult(self):

    class Test(unittest.TestCase):

        def testFoo(self):
            pass
    originalRegisterResult = unittest.runner.registerResult

    def cleanup():
        unittest.runner.registerResult = originalRegisterResult
    self.addCleanup(cleanup)
    result = unittest.TestResult()
    runner = unittest.TextTestRunner(stream=io.StringIO())
    runner._makeResult = lambda: result
    self.wasRegistered = 0

    def fakeRegisterResult(thisResult):
        self.wasRegistered += 1
        self.assertEqual(thisResult, result)
    unittest.runner.registerResult = fakeRegisterResult
    runner.run(unittest.TestSuite())
    self.assertEqual(self.wasRegistered, 1)