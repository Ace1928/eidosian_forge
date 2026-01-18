import unittest
import gc
import sys
import weakref
from unittest.test.support import LoggingResult, TestEquality
def test_init__empty_tests(self):
    suite = unittest.TestSuite([])
    self.assertEqual(suite.countTestCases(), 0)
    suite.run(unittest.TestResult())
    self.assertEqual(suite.countTestCases(), 0)