import warnings
from breezy import symbol_versioning
from breezy.symbol_versioning import (deprecated_function, deprecated_in,
from breezy.tests import TestCase
def test_activate_deprecation_with_DW_error(self):
    warnings.filterwarnings('error', category=DeprecationWarning)
    self.assertFirstWarning('error', DeprecationWarning)
    self.assertEqual(1, len(warnings.filters))
    symbol_versioning.activate_deprecation_warnings(override=False)
    self.assertFirstWarning('error', DeprecationWarning)
    self.assertEqual(1, len(warnings.filters))
    symbol_versioning.activate_deprecation_warnings(override=True)
    self.assertFirstWarning('default', DeprecationWarning)
    self.assertEqual(2, len(warnings.filters))