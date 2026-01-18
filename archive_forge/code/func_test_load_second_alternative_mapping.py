import sys
import types
from heat.engine import plugin_manager
from heat.tests import common
def test_load_second_alternative_mapping(self):
    pm = plugin_manager.PluginMapping(['nonexist', 'current_test'])
    self.assertEqual(current_test_mapping(), pm.load_from_module(self.module()))