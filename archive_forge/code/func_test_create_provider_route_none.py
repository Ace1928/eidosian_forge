from unittest import mock
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.network import nvgreutils
@mock.patch.object(nvgreutils.NvgreUtils, '_get_network_iface_index')
def test_create_provider_route_none(self, mock_get_iface_index):
    mock_get_iface_index.return_value = None
    self.utils.create_provider_route(mock.sentinel.fake_network)
    scimv2 = self.utils._scimv2
    self.assertFalse(scimv2.MSFT_NetVirtualizationProviderRouteSettingData.new.called)