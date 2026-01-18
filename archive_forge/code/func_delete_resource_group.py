from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase, normalize_location_name
def delete_resource_group(self):
    try:
        poller = self.rm_client.resource_groups.begin_delete(self.name)
        self.get_poller_result(poller)
    except Exception as exc:
        self.fail('Error delete resource group {0} - {1}'.format(self.name, str(exc)))
    self.results['state']['status'] = 'Deleted'
    return True