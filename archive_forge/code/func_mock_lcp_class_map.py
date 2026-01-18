from unittest import mock
from heat.common import lifecycle_plugin_utils
from heat.engine import lifecycle_plugin
from heat.engine import resources
from heat.tests import common
def mock_lcp_class_map(self, lcp_mappings):
    self.mock_get_plugins = self.patchobject(resources.global_env(), 'get_stack_lifecycle_plugins', return_value=lcp_mappings)
    lifecycle_plugin_utils.pp_class_instances = None