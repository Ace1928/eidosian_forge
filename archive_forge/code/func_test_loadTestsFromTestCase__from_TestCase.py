import functools
import sys
import types
import warnings
import unittest
def test_loadTestsFromTestCase__from_TestCase(self):
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(unittest.TestCase)
    self.assertIsInstance(suite, loader.suiteClass)
    self.assertEqual(list(suite), [])