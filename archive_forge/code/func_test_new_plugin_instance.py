from unittest import mock
from stevedore import enabled
from neutron_lib.tests import _base as base
from neutron_lib.utils import runtime
@mock.patch.object(enabled, 'EnabledExtensionManager')
def test_new_plugin_instance(self, mock_mgr):
    mock_epa = mock.Mock()
    mock_epa.name = 'a'
    mock_epb = mock.Mock()
    mock_epb.name = 'b'
    mock_mgr().names.return_value = ['a', 'b']
    mock_mgr().map = lambda f: [f(ep) for ep in [mock_epa, mock_epb]]
    plugins = runtime.NamespacedPlugins('_test_ns_')
    plugins.new_plugin_instance('a', 'c', 'd', karg='kval')
    plugins.new_plugin_instance('b')
    mock_epa.plugin.assert_called_once_with('c', 'd', karg='kval')
    mock_epb.plugin.assert_called_once_with()