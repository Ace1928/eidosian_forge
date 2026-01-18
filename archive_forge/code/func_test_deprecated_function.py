import warnings
from breezy import symbol_versioning
from breezy.symbol_versioning import (deprecated_function, deprecated_in,
from breezy.tests import TestCase
def test_deprecated_function(self):
    expected_warning = ('breezy.tests.test_symbol_versioning.sample_deprecated_function was deprecated in version 0.7.0.', DeprecationWarning, 2)
    expected_docstring = 'Deprecated function docstring.\n\nThis function was deprecated in version 0.7.0.\n'
    self.check_deprecated_callable(expected_warning, expected_docstring, 'sample_deprecated_function', 'breezy.tests.test_symbol_versioning', sample_deprecated_function)