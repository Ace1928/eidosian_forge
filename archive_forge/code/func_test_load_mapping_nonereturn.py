import sys
import types
from heat.engine import plugin_manager
from heat.tests import common
def test_load_mapping_nonereturn(self):
    pm = plugin_manager.PluginMapping('none_return_test')
    self.assertEqual({}, pm.load_from_module(self.module()))