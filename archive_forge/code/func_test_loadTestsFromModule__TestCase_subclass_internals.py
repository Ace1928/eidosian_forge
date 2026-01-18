import functools
import sys
import types
import warnings
import unittest
def test_loadTestsFromModule__TestCase_subclass_internals(self):
    m = types.ModuleType('m')
    m.TestCase = unittest.TestCase
    m.FunctionTestCase = unittest.FunctionTestCase
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(m)
    self.assertIsInstance(suite, loader.suiteClass)
    self.assertEqual(list(suite), [])