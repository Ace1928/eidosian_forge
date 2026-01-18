from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase, normalize_location_name
def name_exists(self):
    try:
        exists = self.rm_client.resource_groups.check_existence(self.name)
    except Exception as exc:
        self.fail('Error checking for existence of name {0} - {1}'.format(self.name, str(exc)))
    return exists