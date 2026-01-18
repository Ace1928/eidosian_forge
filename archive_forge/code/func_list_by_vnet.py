from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def list_by_vnet(self):
    """
        Lists the Virtual Network Peering in specific Virtual Network.

        :return: List of Virtual Network Peering
        """
    self.log('List Virtual Network Peering in Virtual Network {0}'.format(self.virtual_network['name']))
    results = []
    try:
        response = self.network_client.virtual_network_peerings.list(resource_group_name=self.resource_group, virtual_network_name=self.virtual_network['name'])
        self.log('Response : {0}'.format(response))
        if response:
            for p in response:
                results.append(vnetpeering_to_dict(p))
    except ResourceNotFoundError:
        self.log('Did not find the Virtual Network Peering.')
    return results