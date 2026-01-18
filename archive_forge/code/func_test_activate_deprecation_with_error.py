import warnings
from breezy import symbol_versioning
from breezy.symbol_versioning import (deprecated_function, deprecated_in,
from breezy.tests import TestCase
def test_activate_deprecation_with_error(self):
    warnings.filterwarnings('error', category=Warning)
    self.assertFirstWarning('error', Warning)
    self.assertEqual(1, len(warnings.filters))
    symbol_versioning.activate_deprecation_warnings(override=False)
    self.assertFirstWarning('error', Warning)
    self.assertEqual(1, len(warnings.filters))