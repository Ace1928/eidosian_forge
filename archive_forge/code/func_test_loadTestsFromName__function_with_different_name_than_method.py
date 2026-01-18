import functools
import sys
import types
import warnings
import unittest
def test_loadTestsFromName__function_with_different_name_than_method(self):
    m = types.ModuleType('m')

    class MyTestCase(unittest.TestCase):
        test = lambda: 1
    m.testcase_1 = MyTestCase
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromNames(['testcase_1.test'], m)
    self.assertIsInstance(suite, loader.suiteClass)
    ref_suite = unittest.TestSuite([MyTestCase('test')])
    self.assertEqual(list(suite), [ref_suite])