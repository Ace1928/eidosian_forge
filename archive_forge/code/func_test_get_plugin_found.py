from neutron_lib.plugins import directory
from neutron_lib.tests import _base as base
def test_get_plugin_found(self):
    self.plugin_directory._plugins = {'foo': lambda *x, **y: 'bar'}
    plugin = self.plugin_directory.get_plugin('foo')
    self.assertEqual('bar', plugin())