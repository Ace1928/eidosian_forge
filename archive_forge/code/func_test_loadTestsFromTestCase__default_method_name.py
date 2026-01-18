import functools
import sys
import types
import warnings
import unittest
def test_loadTestsFromTestCase__default_method_name(self):

    class Foo(unittest.TestCase):

        def runTest(self):
            pass
    loader = unittest.TestLoader()
    self.assertFalse('runTest'.startswith(loader.testMethodPrefix))
    suite = loader.loadTestsFromTestCase(Foo)
    self.assertIsInstance(suite, loader.suiteClass)
    self.assertEqual(list(suite), [Foo('runTest')])