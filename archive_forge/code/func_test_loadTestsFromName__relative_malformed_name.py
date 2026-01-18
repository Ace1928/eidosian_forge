import functools
import sys
import types
import warnings
import unittest
def test_loadTestsFromName__relative_malformed_name(self):
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromName('abc () //', unittest)
    error, test = self.check_deferred_error(loader, suite)
    expected = "module 'unittest' has no attribute 'abc () //'"
    expected_regex = "module 'unittest' has no attribute 'abc \\(\\) //'"
    self.assertIn(expected, error, 'missing error string in %r' % error)
    self.assertRaisesRegex(AttributeError, expected_regex, getattr(test, 'abc () //'))