import unittest
import gc
import sys
import weakref
from unittest.test.support import LoggingResult, TestEquality
def test_init__TestSuite_instances_in_tests(self):

    def tests():
        ftc = unittest.FunctionTestCase(lambda: None)
        yield unittest.TestSuite([ftc])
        yield unittest.FunctionTestCase(lambda: None)
    suite = unittest.TestSuite(tests())
    self.assertEqual(suite.countTestCases(), 2)
    suite.run(unittest.TestResult())
    self.assertEqual(suite.countTestCases(), 2)