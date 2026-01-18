from unittest import mock
from stevedore import enabled
from neutron_lib.tests import _base as base
from neutron_lib.utils import runtime
@mock.patch.object(enabled, 'EnabledExtensionManager')
def test_init_reload(self, mock_mgr):
    plugins = runtime.NamespacedPlugins('_test_ns_')
    mock_mgr.assert_called_with('_test_ns_', mock.ANY, invoke_on_load=False)
    mock_mgr().map.assert_called_with(plugins._add_extension)