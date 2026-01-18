from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common_ext import AzureRMModuleBaseExt
def list_by_resourcegroup(self):
    self.log('List all Batch Account in the rsource group {0}'.format(self.resource_group))
    result = []
    response = []
    try:
        response = self.mgmt_client.batch_account.list_by_resource_group(resource_group_name=self.resource_group)
        self.log('Response : {0}'.format(response))
    except Exception as e:
        self.log('Did not find the Batch Account instance. Exception as {0}'.format(e))
    for item in response:
        result.append(item.as_dict())
    return result