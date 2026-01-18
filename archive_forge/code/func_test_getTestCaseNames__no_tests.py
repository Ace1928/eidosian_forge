import functools
import sys
import types
import warnings
import unittest
def test_getTestCaseNames__no_tests(self):

    class Test(unittest.TestCase):

        def foobar(self):
            pass
    loader = unittest.TestLoader()
    self.assertEqual(loader.getTestCaseNames(Test), [])