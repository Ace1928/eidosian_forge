from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
from ansible.module_utils._text import to_native
def get_blob_mgmt_props(self, resource_group, name):
    if not self.show_blob_cors:
        return None
    try:
        return self.storage_client.blob_services.get_service_properties(resource_group, name)
    except Exception:
        pass
    return None