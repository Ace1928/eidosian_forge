import warnings
from breezy import symbol_versioning
from breezy.symbol_versioning import (deprecated_function, deprecated_in,
from breezy.tests import TestCase
def test_deprecated_dict(self):
    expected_warning = ('access to a_deprecated_dict was deprecated in version 0.14.0. Pull the other one!', DeprecationWarning, 2)
    old_warning_method = symbol_versioning.warn
    try:
        symbol_versioning.set_warning_method(self.capture_warning)
        self.assertEqual(len(a_deprecated_dict), 1)
        self.assertEqual([expected_warning], self._warnings)
        a_deprecated_dict['b'] = 42
        self.assertEqual(a_deprecated_dict['b'], 42)
        self.assertTrue('b' in a_deprecated_dict)
        del a_deprecated_dict['b']
        self.assertFalse('b' in a_deprecated_dict)
        self.assertEqual([expected_warning] * 6, self._warnings)
    finally:
        symbol_versioning.set_warning_method(old_warning_method)