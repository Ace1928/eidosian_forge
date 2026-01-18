import functools
import sys
import types
import warnings
import unittest
def test_loadTestsFromName__module_not_loaded(self):
    module_name = 'unittest.test.dummy'
    sys.modules.pop(module_name, None)
    loader = unittest.TestLoader()
    try:
        suite = loader.loadTestsFromName(module_name)
        self.assertIsInstance(suite, loader.suiteClass)
        self.assertEqual(list(suite), [])
        self.assertIn(module_name, sys.modules)
    finally:
        if module_name in sys.modules:
            del sys.modules[module_name]