import unittest
import gc
import sys
import weakref
from unittest.test.support import LoggingResult, TestEquality
def test_addTest__TestCase(self):

    class Foo(unittest.TestCase):

        def test(self):
            pass
    test = Foo('test')
    suite = unittest.TestSuite()
    suite.addTest(test)
    self.assertEqual(suite.countTestCases(), 1)
    self.assertEqual(list(suite), [test])
    suite.run(unittest.TestResult())
    self.assertEqual(suite.countTestCases(), 1)