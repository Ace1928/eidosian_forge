from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase, format_resource_id
def create_data_disks(self):
    return list(filter(None, [self.create_data_disk(lun, source) for lun, source in enumerate(self.data_disk_sources)]))