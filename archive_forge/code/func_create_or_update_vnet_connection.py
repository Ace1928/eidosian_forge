from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def create_or_update_vnet_connection(self, vnet):
    try:
        return self.web_client.web_apps.create_or_update_swift_virtual_network_connection_with_check(resource_group_name=self.resource_group, name=self.name, connection_envelope=vnet)
    except Exception as exc:
        self.fail('Error creating/updating webapp vnet connection {0} (vnet={1}, rg={2}) - {3}'.format(self.name, self.vnet_name, self.resource_group, str(exc)))