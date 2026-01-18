from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
from ansible.module_utils._text import to_native
def get_blob_client_props(self, resource_group, name, kind):
    if kind == 'FileStorage':
        return None
    try:
        return self.get_blob_service_client(resource_group, name).get_service_properties()
    except Exception:
        pass
    return None