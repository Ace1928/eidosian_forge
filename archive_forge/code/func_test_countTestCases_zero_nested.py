import unittest
import gc
import sys
import weakref
from unittest.test.support import LoggingResult, TestEquality
def test_countTestCases_zero_nested(self):

    class Test1(unittest.TestCase):

        def test(self):
            pass
    suite = unittest.TestSuite([unittest.TestSuite()])
    self.assertEqual(suite.countTestCases(), 0)