from neutron_lib.plugins import directory
from neutron_lib.tests import _base as base
def test__get_plugin_directory(self):
    self.assertIsNotNone(directory._get_plugin_directory())