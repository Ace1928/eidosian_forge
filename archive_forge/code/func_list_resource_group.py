from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def list_resource_group(self):
    self.log('List items for resource group')
    try:
        response = self.private_dns_client.private_zones.list_by_resource_group(self.resource_group)
    except Exception as exc:
        self.fail('Failed to list for resource group {0} - {1}'.format(self.resource_group, str(exc)))
    results = []
    for item in response:
        if self.has_tags(item.tags, self.tags):
            results.append(item)
    return results