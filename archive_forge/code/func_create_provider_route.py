from oslo_log import log as logging
from os_win import constants
from os_win import exceptions
from os_win.utils import baseutils
from os_win.utils.network import networkutils
def create_provider_route(self, network_name):
    iface_index = self._get_network_iface_index(network_name)
    routes = self._scimv2.MSFT_NetVirtualizationProviderRouteSettingData(InterfaceIndex=iface_index, NextHop=constants.IPV4_DEFAULT)
    if not routes:
        self._create_new_object(self._scimv2.MSFT_NetVirtualizationProviderRouteSettingData, InterfaceIndex=iface_index, DestinationPrefix='%s/0' % constants.IPV4_DEFAULT, NextHop=constants.IPV4_DEFAULT)