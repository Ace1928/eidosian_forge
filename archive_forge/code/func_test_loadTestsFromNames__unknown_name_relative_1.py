import functools
import sys
import types
import warnings
import unittest
def test_loadTestsFromNames__unknown_name_relative_1(self):
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromNames(['sdasfasfasdf'], unittest)
    error, test = self.check_deferred_error(loader, list(suite)[0])
    expected = "module 'unittest' has no attribute 'sdasfasfasdf'"
    self.assertIn(expected, error, 'missing error string in %r' % error)
    self.assertRaisesRegex(AttributeError, expected, test.sdasfasfasdf)