import sys
from .. import plugin as _mod_plugin
from .. import symbol_versioning, tests
from . import features
def test_available_plugin(self):
    plugins = _mod_plugin.plugins()
    if not plugins:
        self.skipTest('no plugins available to test with')
    a_plugin_name = next(iter(plugins))
    feature = features.PluginLoadedFeature(a_plugin_name)
    self.assertEqual(a_plugin_name, feature.plugin_name)
    self.assertEqual(a_plugin_name + ' plugin', str(feature))
    self.assertTrue(feature.available())