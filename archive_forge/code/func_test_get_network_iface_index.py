from unittest import mock
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.network import nvgreutils
def test_get_network_iface_index(self):
    fake_network = mock.MagicMock(InterfaceIndex=mock.sentinel.iface_index)
    self.utils._scimv2.MSFT_NetAdapter.return_value = [fake_network]
    description = self.utils._utils.get_vswitch_external_network_name.return_value
    index = self.utils._get_network_iface_index(mock.sentinel.fake_network)
    self.assertEqual(mock.sentinel.iface_index, index)
    self.assertIn(mock.sentinel.fake_network, self.utils._net_if_indexes)
    self.utils._scimv2.MSFT_NetAdapter.assert_called_once_with(InterfaceDescription=description)