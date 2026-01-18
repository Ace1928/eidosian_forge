from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def list_disks(self):
    """Get all managed disks"""
    results = []
    try:
        results = self.compute_client.disks.list()
        if self.managed_by:
            results = [disk for disk in results if disk.managed_by == self.managed_by]
        if self.tags:
            results = [disk for disk in results if self.has_tags(disk.tags, self.tags)]
        results = [self.managed_disk_to_dict(disk) for disk in results]
    except ResourceNotFoundError as exc:
        self.fail('Failed to list all items - {0}'.format(str(exc)))
    return results