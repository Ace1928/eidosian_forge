import functools
import sys
import types
import warnings
import unittest
def test_loadTestsFromNames__relative_empty_name(self):
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromNames([''], unittest)
    error, test = self.check_deferred_error(loader, list(suite)[0])
    expected = "has no attribute ''"
    self.assertIn(expected, error, 'missing error string in %r' % error)
    self.assertRaisesRegex(AttributeError, expected, getattr(test, ''))