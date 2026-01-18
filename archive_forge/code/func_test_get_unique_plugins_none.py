from neutron_lib.plugins import directory
from neutron_lib.tests import _base as base
def test_get_unique_plugins_none(self):
    self.assertFalse(directory.get_unique_plugins())