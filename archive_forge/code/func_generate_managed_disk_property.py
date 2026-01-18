from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def generate_managed_disk_property(self):
    disk_params = {}
    creation_data = {}
    disk_params['location'] = self.location
    disk_params['tags'] = self.tags
    if self.zone:
        disk_params['zones'] = [self.zone]
    if self.storage_account_type:
        storage_account_type = self.compute_models.DiskSku(name=self.storage_account_type)
        disk_params['sku'] = storage_account_type
    disk_params['disk_size_gb'] = self.disk_size_gb
    creation_data['create_option'] = self.compute_models.DiskCreateOption.empty
    if self.create_option == 'import':
        creation_data['create_option'] = self.compute_models.DiskCreateOption.import_enum
        creation_data['source_uri'] = self.source_uri
        creation_data['storage_account_id'] = self.storage_account_id
    elif self.create_option == 'copy':
        creation_data['create_option'] = self.compute_models.DiskCreateOption.copy
        creation_data['source_resource_id'] = self.source_uri
    if self.os_type:
        disk_params['os_type'] = self.compute_models.OperatingSystemTypes(self.os_type.capitalize())
    else:
        disk_params['os_type'] = None
    if self.max_shares:
        disk_params['max_shares'] = self.max_shares
    disk_params['creation_data'] = creation_data
    return disk_params