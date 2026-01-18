from unittest import mock
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.network import nvgreutils
def test_clear_customer_routes(self):
    cls = self.utils._scimv2.MSFT_NetVirtualizationCustomerRouteSettingData
    route = mock.MagicMock()
    cls.return_value = [route]
    self.utils.clear_customer_routes(mock.sentinel.vsid)
    cls.assert_called_once_with(VirtualSubnetID=mock.sentinel.vsid)
    route.Delete_.assert_called_once_with()