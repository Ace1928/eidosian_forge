import warnings
from breezy import symbol_versioning
from breezy.symbol_versioning import (deprecated_function, deprecated_in,
from breezy.tests import TestCase
def test_deprecation_string(self):
    """We can get a deprecation string for a method or function."""
    err_message = symbol_versioning.deprecation_string(self.test_deprecation_string, deprecated_in((0, 11, 0)))
    self.assertEqual(err_message, 'breezy.tests.test_symbol_versioning.TestDeprecationWarnings.test_deprecation_string was deprecated in version 0.11.0.')
    self.assertEqual('breezy.symbol_versioning.deprecated_function was deprecated in version 0.11.0.', symbol_versioning.deprecation_string(symbol_versioning.deprecated_function, deprecated_in((0, 11, 0))))