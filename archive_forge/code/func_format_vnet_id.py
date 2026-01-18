from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase, format_resource_id
def format_vnet_id(self, vnet):
    if not vnet:
        return vnet
    if isinstance(vnet, dict) and vnet.get('name') and vnet.get('resource_group'):
        remote_vnet_id = format_resource_id(vnet['name'], self.subscription_id, 'Microsoft.Network', 'virtualNetworks', vnet['resource_group'])
    elif isinstance(vnet, str):
        if is_valid_resource_id(vnet):
            remote_vnet_id = vnet
        else:
            remote_vnet_id = format_resource_id(vnet, self.subscription_id, 'Microsoft.Network', 'virtualNetworks', self.resource_group)
    else:
        self.fail('remote_virtual_network could be a valid resource id, dict of name and resource_group, name of virtual network in same resource group.')
    return remote_vnet_id