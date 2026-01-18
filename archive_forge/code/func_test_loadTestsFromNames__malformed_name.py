import functools
import sys
import types
import warnings
import unittest
def test_loadTestsFromNames__malformed_name(self):
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromNames(['abc () //'])
    error, test = self.check_deferred_error(loader, list(suite)[0])
    expected = 'Failed to import test module: abc () //'
    expected_regex = 'Failed to import test module: abc \\(\\) //'
    self.assertIn(expected, error, 'missing error string in %r' % error)
    self.assertRaisesRegex(ImportError, expected_regex, getattr(test, 'abc () //'))