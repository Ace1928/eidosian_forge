import sys
import types
from heat.engine import plugin_manager
from heat.tests import common
def test_load_mapping_non_existent(self):
    pm = plugin_manager.PluginMapping('nonexist')
    self.assertEqual({}, pm.load_from_module(self.module()))