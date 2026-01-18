from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMAuth
def get_managed_by_tenants_list(self, object_list):
    return [dict(tenantId=item.tenant_id) for item in object_list]