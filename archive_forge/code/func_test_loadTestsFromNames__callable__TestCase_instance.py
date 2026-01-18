import functools
import sys
import types
import warnings
import unittest
def test_loadTestsFromNames__callable__TestCase_instance(self):
    m = types.ModuleType('m')
    testcase_1 = unittest.FunctionTestCase(lambda: None)

    def return_TestCase():
        return testcase_1
    m.return_TestCase = return_TestCase
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromNames(['return_TestCase'], m)
    self.assertIsInstance(suite, loader.suiteClass)
    ref_suite = unittest.TestSuite([testcase_1])
    self.assertEqual(list(suite), [ref_suite])