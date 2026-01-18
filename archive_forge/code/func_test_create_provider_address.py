from unittest import mock
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.network import nvgreutils
@mock.patch.object(nvgreutils.NvgreUtils, 'get_network_iface_ip')
@mock.patch.object(nvgreutils.NvgreUtils, '_get_network_iface_index')
def test_create_provider_address(self, mock_get_iface_index, mock_get_iface_ip):
    mock_get_iface_index.return_value = mock.sentinel.iface_index
    mock_get_iface_ip.return_value = (mock.sentinel.iface_ip, mock.sentinel.prefix_len)
    provider_addr = mock.MagicMock()
    scimv2 = self.utils._scimv2
    obj_class = scimv2.MSFT_NetVirtualizationProviderAddressSettingData
    obj_class.return_value = [provider_addr]
    self.utils.create_provider_address(mock.sentinel.fake_network, mock.sentinel.fake_vlan_id)
    self.assertTrue(provider_addr.Delete_.called)
    obj_class.new.assert_called_once_with(ProviderAddress=mock.sentinel.iface_ip, VlanID=mock.sentinel.fake_vlan_id, InterfaceIndex=mock.sentinel.iface_index, PrefixLength=mock.sentinel.prefix_len)