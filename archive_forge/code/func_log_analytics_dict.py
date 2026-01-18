from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common_ext import AzureRMModuleBaseExt
def log_analytics_dict(self, workspace_id):
    if workspace_id:
        return dict(id=workspace_id)
    return None