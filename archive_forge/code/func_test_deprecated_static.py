import warnings
from breezy import symbol_versioning
from breezy.symbol_versioning import (deprecated_function, deprecated_in,
from breezy.tests import TestCase
def test_deprecated_static(self):
    expected_warning = ('breezy.tests.test_symbol_versioning.TestDeprecationWarnings.deprecated_static was deprecated in version 0.7.0.', DeprecationWarning, 2)
    expected_docstring = 'Deprecated static.\n\nThis function was deprecated in version 0.7.0.\n'
    self.check_deprecated_callable(expected_warning, expected_docstring, 'deprecated_static', 'breezy.tests.test_symbol_versioning', self.deprecated_static)