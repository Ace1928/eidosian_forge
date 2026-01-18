import functools
import sys
import types
import warnings
import unittest
def test_loadTestsFromNames__relative_testmethod(self):
    m = types.ModuleType('m')

    class MyTestCase(unittest.TestCase):

        def test(self):
            pass
    m.testcase_1 = MyTestCase
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromNames(['testcase_1.test'], m)
    self.assertIsInstance(suite, loader.suiteClass)
    ref_suite = unittest.TestSuite([MyTestCase('test')])
    self.assertEqual(list(suite), [ref_suite])