from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def get_managed_disk(self, resource_group, name):
    try:
        resp = self.compute_client.disks.get(resource_group, name)
        return managed_disk_to_dict(resp)
    except ResourceNotFoundError:
        self.log('Did not find managed disk {0}/{1}'.format(resource_group, name))