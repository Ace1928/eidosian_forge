from neutron_lib.plugins import directory
from neutron_lib.tests import _base as base
def test_add_plugin(self):
    self.plugin_directory.add_plugin('foo', 'bar')
    self.assertEqual(1, len(self.plugin_directory._plugins))