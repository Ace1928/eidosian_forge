from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase, format_resource_id
def sync_vnet_peering(self):
    """
        Creates or Update Azure Virtual Network Peering.

        :return: deserialized Azure Virtual Network Peering instance state dictionary
        """
    self.log('Creating or Updating the Azure Virtual Network Peering {0}'.format(self.name))
    vnet_id = format_resource_id(self.virtual_network['name'], self.subscription_id, 'Microsoft.Network', 'virtualNetworks', self.virtual_network['resource_group'])
    peering = self.network_models.VirtualNetworkPeering(id=vnet_id, name=self.name, remote_virtual_network=self.network_models.SubResource(id=self.remote_virtual_network), allow_virtual_network_access=self.allow_virtual_network_access, allow_gateway_transit=self.allow_gateway_transit, allow_forwarded_traffic=self.allow_forwarded_traffic, use_remote_gateways=self.use_remote_gateways)
    try:
        response = self.network_client.virtual_network_peerings.begin_create_or_update(self.resource_group, self.virtual_network['name'], self.name, peering, sync_remote_address_space=True)
        if isinstance(response, LROPoller):
            response = self.get_poller_result(response)
        return vnetpeering_to_dict(response)
    except Exception as exc:
        self.fail('Error creating Azure Virtual Network Peering: {0}.'.format(exc.message))