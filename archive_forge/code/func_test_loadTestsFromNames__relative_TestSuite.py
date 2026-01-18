import functools
import sys
import types
import warnings
import unittest
def test_loadTestsFromNames__relative_TestSuite(self):
    m = types.ModuleType('m')

    class MyTestCase(unittest.TestCase):

        def test(self):
            pass
    m.testsuite = unittest.TestSuite([MyTestCase('test')])
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromNames(['testsuite'], m)
    self.assertIsInstance(suite, loader.suiteClass)
    self.assertEqual(list(suite), [m.testsuite])