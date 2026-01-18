import functools
import sys
import types
import warnings
import unittest
def test_loadTestsFromTestCase__from_FunctionTestCase(self):
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(unittest.FunctionTestCase)
    self.assertIsInstance(suite, loader.suiteClass)
    self.assertEqual(list(suite), [])