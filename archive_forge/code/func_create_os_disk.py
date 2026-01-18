from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase, format_resource_id
def create_os_disk(self):
    blob_uri, disk, snapshot = self.resolve_storage_source(self.source)
    snapshot_resource = self.image_models.SubResource(id=snapshot) if snapshot else None
    managed_disk = self.image_models.SubResource(id=disk) if disk else None
    return self.image_models.ImageOSDisk(os_type=self.os_type, os_state=self.image_models.OperatingSystemStateTypes.generalized, snapshot=snapshot_resource, managed_disk=managed_disk, blob_uri=blob_uri)