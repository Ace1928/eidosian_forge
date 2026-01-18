from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase, format_resource_id
def get_vnet(self, resource_group, vnet_name):
    """
        Get Azure Virtual Network
        :return: deserialized Azure Virtual Network
        """
    self.log('Get the Azure Virtual Network {0}'.format(vnet_name))
    vnet = self.network_client.virtual_networks.get(resource_group, vnet_name)
    if vnet:
        results = virtual_network_to_dict(vnet)
        return results
    return False