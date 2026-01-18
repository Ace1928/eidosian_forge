from unittest import mock
import oslotest.base as base
from osc_placement import version
def test_compare_with_exc(self):
    self.assertTrue(version.compare('1.05', version.gt('1.4')))
    self.assertFalse(version.compare('1.3', version.gt('1.4'), exc=False))
    self.assertRaisesRegex(ValueError, 'Operation or argument is not supported', version.compare, '3.1.2', version.gt('3.1.3'))