import functools
import sys
import types
import warnings
import unittest
def test_loadTestsFromModule__TestCase_subclass(self):
    m = types.ModuleType('m')

    class MyTestCase(unittest.TestCase):

        def test(self):
            pass
    m.testcase_1 = MyTestCase
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(m)
    self.assertIsInstance(suite, loader.suiteClass)
    expected = [loader.suiteClass([MyTestCase('test')])]
    self.assertEqual(list(suite), expected)