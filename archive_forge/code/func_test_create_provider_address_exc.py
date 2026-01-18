from unittest import mock
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.network import nvgreutils
@mock.patch.object(nvgreutils.NvgreUtils, 'get_network_iface_ip')
@mock.patch.object(nvgreutils.NvgreUtils, '_get_network_iface_index')
def test_create_provider_address_exc(self, mock_get_iface_index, mock_get_iface_ip):
    mock_get_iface_ip.return_value = (None, None)
    self.assertRaises(exceptions.NotFound, self.utils.create_provider_address, mock.sentinel.fake_network, mock.sentinel.fake_vlan_id)