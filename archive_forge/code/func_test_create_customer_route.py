from unittest import mock
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.network import nvgreutils
def test_create_customer_route(self):
    self.utils.create_customer_route(mock.sentinel.fake_vsid, mock.sentinel.dest_prefix, mock.sentinel.next_hop, self._FAKE_RDID)
    scimv2 = self.utils._scimv2
    obj_class = scimv2.MSFT_NetVirtualizationCustomerRouteSettingData
    obj_class.new.assert_called_once_with(VirtualSubnetID=mock.sentinel.fake_vsid, DestinationPrefix=mock.sentinel.dest_prefix, NextHop=mock.sentinel.next_hop, Metric=255, RoutingDomainID='{%s}' % self._FAKE_RDID)