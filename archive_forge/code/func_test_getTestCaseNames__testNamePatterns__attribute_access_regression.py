import functools
import sys
import types
import warnings
import unittest
def test_getTestCaseNames__testNamePatterns__attribute_access_regression(self):

    class Trap:

        def __get__(*ignored):
            self.fail('Non-test attribute accessed')

    class MyTest(unittest.TestCase):

        def test_1(self):
            pass
        foobar = Trap()
    loader = unittest.TestLoader()
    self.assertEqual(loader.getTestCaseNames(MyTest), ['test_1'])
    loader = unittest.TestLoader()
    loader.testNamePatterns = []
    self.assertEqual(loader.getTestCaseNames(MyTest), [])