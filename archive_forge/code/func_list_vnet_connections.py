from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def list_vnet_connections(self):
    try:
        return self.web_client.web_apps.list_vnet_connections(resource_group_name=self.resource_group, name=self.name)
    except Exception as exc:
        self.fail('Error getting webapp vnet connections {0} (rg={1}) - {2}'.format(self.name, self.resource_group, str(exc)))