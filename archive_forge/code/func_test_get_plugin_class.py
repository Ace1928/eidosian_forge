from unittest import mock
from stevedore import enabled
from neutron_lib.tests import _base as base
from neutron_lib.utils import runtime
@mock.patch.object(enabled, 'EnabledExtensionManager')
def test_get_plugin_class(self, mock_mgr):
    mock_epa = mock.Mock()
    mock_epa.name = 'a'
    mock_epa.plugin = 'A'
    mock_epb = mock.Mock()
    mock_epb.name = 'b'
    mock_epb.plugin = 'B'
    mock_mgr().names.return_value = ['a', 'b']
    mock_mgr().map = lambda f: [f(ep) for ep in [mock_epa, mock_epb]]
    plugins = runtime.NamespacedPlugins('_test_ns_')
    self.assertEqual('A', plugins.get_plugin_class('a'))
    self.assertEqual('B', plugins.get_plugin_class('b'))