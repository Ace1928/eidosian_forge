from unittest import mock
from stevedore import enabled
from neutron_lib.tests import _base as base
from neutron_lib.utils import runtime
@mock.patch.object(enabled, 'EnabledExtensionManager')
def test_add_duplicate_names(self, mock_mgr):
    mock_ep = mock.Mock()
    mock_ep.name = 'a'
    mock_mgr().names.return_value = ['a', 'a']
    mock_mgr().map = lambda f: [f(ep) for ep in [mock_ep, mock_ep]]
    self.assertRaises(KeyError, runtime.NamespacedPlugins, '_test_ns_')