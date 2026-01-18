import functools
import sys
import types
import warnings
import unittest
def test_loadTestsFromName__unknown_module_name(self):
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromName('sdasfasfasdf')
    expected = "No module named 'sdasfasfasdf'"
    error, test = self.check_deferred_error(loader, suite)
    self.assertIn(expected, error, 'missing error string in %r' % error)
    self.assertRaisesRegex(ImportError, expected, test.sdasfasfasdf)