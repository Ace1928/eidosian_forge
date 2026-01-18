import sys
import types
from heat.engine import plugin_manager
from heat.tests import common
def test_load_mapping_exception(self):
    pm = plugin_manager.PluginMapping('error_test_exception')
    self.assertRaisesRegex(Exception, 'exception', pm.load_from_module, self.module())