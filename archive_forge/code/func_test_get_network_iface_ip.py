from unittest import mock
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.network import nvgreutils
@mock.patch.object(nvgreutils.NvgreUtils, '_get_network_ifaces_by_name')
def test_get_network_iface_ip(self, mock_get_net_ifaces):
    fake_network = mock.MagicMock(InterfaceIndex=mock.sentinel.iface_index, DriverDescription=self.utils._HYPERV_VIRT_ADAPTER)
    mock_get_net_ifaces.return_value = [fake_network]
    fake_netip = mock.MagicMock(IPAddress=mock.sentinel.provider_addr, PrefixLength=mock.sentinel.prefix_len)
    self.utils._scimv2.MSFT_NetIPAddress.return_value = [fake_netip]
    pair = self.utils.get_network_iface_ip(mock.sentinel.fake_network)
    self.assertEqual((mock.sentinel.provider_addr, mock.sentinel.prefix_len), pair)