import os.path
from os.path import abspath
import re
import sys
import types
import pickle
from test import support
from test.support import import_helper
import unittest
import unittest.mock
import unittest.test
def test_discover_with_modules_that_fail_to_import(self):
    loader = unittest.TestLoader()
    self.setup_import_issue_tests('test_this_does_not_exist.py')
    suite = loader.discover('.')
    self.assertIn(os.getcwd(), sys.path)
    self.assertEqual(suite.countTestCases(), 1)
    self.assertNotEqual([], loader.errors)
    self.assertEqual(1, len(loader.errors))
    error = loader.errors[0]
    self.assertTrue('Failed to import test module: test_this_does_not_exist' in error, 'missing error string in %r' % error)
    test = list(list(suite)[0])[0]
    with self.assertRaises(ImportError):
        test.test_this_does_not_exist()