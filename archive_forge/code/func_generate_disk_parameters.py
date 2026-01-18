from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def generate_disk_parameters(self, location, tags, zone=None, storage_account_type=None, disk_size_gb=None, create_option=None, source_uri=None, storage_account_id=None, os_type=None, max_shares=None, **kwargs):
    disk_params = {}
    creation_data = {}
    disk_params['location'] = location
    disk_params['tags'] = tags
    if zone:
        disk_params['zones'] = [zone]
    if storage_account_type:
        storage = self.compute_models.DiskSku(name=storage_account_type)
        disk_params['sku'] = storage
    disk_params['disk_size_gb'] = disk_size_gb
    creation_data['create_option'] = self.compute_models.DiskCreateOption.empty
    if create_option == 'import':
        creation_data['create_option'] = self.compute_models.DiskCreateOption.import_enum
        creation_data['source_uri'] = source_uri
        creation_data['storage_account_id'] = storage_account_id
    elif create_option == 'copy':
        creation_data['create_option'] = self.compute_models.DiskCreateOption.copy
        creation_data['source_resource_id'] = source_uri
    if os_type:
        disk_params['os_type'] = self.compute_models.OperatingSystemTypes(os_type.capitalize())
    else:
        disk_params['os_type'] = None
    if max_shares:
        disk_params['max_shares'] = max_shares
    disk_params['creation_data'] = creation_data
    return disk_params