import functools
import sys
import types
import warnings
import unittest
def test_getTestCaseNames__not_a_TestCase(self):

    class BadCase(int):

        def test_foo(self):
            pass
    loader = unittest.TestLoader()
    names = loader.getTestCaseNames(BadCase)
    self.assertEqual(names, ['test_foo'])