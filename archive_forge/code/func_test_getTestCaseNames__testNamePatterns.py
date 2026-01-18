import functools
import sys
import types
import warnings
import unittest
def test_getTestCaseNames__testNamePatterns(self):

    class MyTest(unittest.TestCase):

        def test_1(self):
            pass

        def test_2(self):
            pass

        def foobar(self):
            pass
    loader = unittest.TestLoader()
    loader.testNamePatterns = []
    self.assertEqual(loader.getTestCaseNames(MyTest), [])
    loader.testNamePatterns = ['*1']
    self.assertEqual(loader.getTestCaseNames(MyTest), ['test_1'])
    loader.testNamePatterns = ['*1', '*2']
    self.assertEqual(loader.getTestCaseNames(MyTest), ['test_1', 'test_2'])
    loader.testNamePatterns = ['*My*']
    self.assertEqual(loader.getTestCaseNames(MyTest), ['test_1', 'test_2'])
    loader.testNamePatterns = ['*my*']
    self.assertEqual(loader.getTestCaseNames(MyTest), [])