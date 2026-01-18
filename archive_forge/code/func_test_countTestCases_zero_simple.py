import unittest
import gc
import sys
import weakref
from unittest.test.support import LoggingResult, TestEquality
def test_countTestCases_zero_simple(self):
    suite = unittest.TestSuite()
    self.assertEqual(suite.countTestCases(), 0)