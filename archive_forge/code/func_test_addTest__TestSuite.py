import unittest
import gc
import sys
import weakref
from unittest.test.support import LoggingResult, TestEquality
def test_addTest__TestSuite(self):

    class Foo(unittest.TestCase):

        def test(self):
            pass
    suite_2 = unittest.TestSuite([Foo('test')])
    suite = unittest.TestSuite()
    suite.addTest(suite_2)
    self.assertEqual(suite.countTestCases(), 1)
    self.assertEqual(list(suite), [suite_2])
    suite.run(unittest.TestResult())
    self.assertEqual(suite.countTestCases(), 1)