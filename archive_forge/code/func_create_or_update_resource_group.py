from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase, normalize_location_name
def create_or_update_resource_group(self, params):
    try:
        result = self.rm_client.resource_groups.create_or_update(self.name, params)
    except Exception as exc:
        self.fail('Error creating or updating resource group {0} - {1}'.format(self.name, str(exc)))
    return resource_group_to_dict(result)