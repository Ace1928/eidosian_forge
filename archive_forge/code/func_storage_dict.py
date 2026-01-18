from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common_ext import AzureRMModuleBaseExt
def storage_dict(self, storage_account_id):
    if storage_account_id:
        return dict(id=storage_account_id)
    return None