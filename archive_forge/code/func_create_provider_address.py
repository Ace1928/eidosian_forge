from oslo_log import log as logging
from os_win import constants
from os_win import exceptions
from os_win.utils import baseutils
from os_win.utils.network import networkutils
def create_provider_address(self, network_name, provider_vlan_id):
    iface_index = self._get_network_iface_index(network_name)
    provider_addr, prefix_len = self.get_network_iface_ip(network_name)
    if not provider_addr:
        raise exceptions.NotFound(resource=network_name)
    provider = self._scimv2.MSFT_NetVirtualizationProviderAddressSettingData(ProviderAddress=provider_addr)
    if provider:
        if provider[0].VlanID == provider_vlan_id and provider[0].InterfaceIndex == iface_index:
            return
        provider[0].Delete_()
    self._create_new_object(self._scimv2.MSFT_NetVirtualizationProviderAddressSettingData, ProviderAddress=provider_addr, VlanID=provider_vlan_id, InterfaceIndex=iface_index, PrefixLength=prefix_len)