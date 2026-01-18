import unittest
import gc
import sys
import weakref
from unittest.test.support import LoggingResult, TestEquality
def test_countTestCases_simple(self):
    test1 = unittest.FunctionTestCase(lambda: None)
    test2 = unittest.FunctionTestCase(lambda: None)
    suite = unittest.TestSuite((test1, test2))
    self.assertEqual(suite.countTestCases(), 2)
    suite.run(unittest.TestResult())
    self.assertEqual(suite.countTestCases(), 2)