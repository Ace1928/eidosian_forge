from oslo_log import log as logging
from os_win import constants
from os_win import exceptions
from os_win.utils import baseutils
from os_win.utils.network import networkutils
def clear_customer_routes(self, vsid):
    routes = self._scimv2.MSFT_NetVirtualizationCustomerRouteSettingData(VirtualSubnetID=vsid)
    for route in routes:
        route.Delete_()