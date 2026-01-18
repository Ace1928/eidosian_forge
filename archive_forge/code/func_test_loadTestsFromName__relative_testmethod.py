import functools
import sys
import types
import warnings
import unittest
def test_loadTestsFromName__relative_testmethod(self):
    m = types.ModuleType('m')

    class MyTestCase(unittest.TestCase):

        def test(self):
            pass
    m.testcase_1 = MyTestCase
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromName('testcase_1.test', m)
    self.assertIsInstance(suite, loader.suiteClass)
    self.assertEqual(list(suite), [MyTestCase('test')])