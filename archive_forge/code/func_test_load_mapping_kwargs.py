import sys
import types
from heat.engine import plugin_manager
from heat.tests import common
def test_load_mapping_kwargs(self):
    pm = plugin_manager.PluginMapping('kwargs_test', baz='quux')
    self.assertEqual({'baz': 'quux'}, pm.load_from_module(self.module()))