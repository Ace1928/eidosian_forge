from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase, format_resource_id
from ansible.module_utils._text import to_native
def preprocess_vn_list(self, vn_list):
    return [self.parse_vn_id(x) for x in vn_list] if vn_list else None