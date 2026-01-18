from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def delete_managed_disk(self):
    try:
        poller = self.compute_client.disks.begin_delete(self.resource_group, self.name)
        return self.get_poller_result(poller)
    except Exception as e:
        self.fail('Error deleting the managed disk: {0}'.format(str(e)))