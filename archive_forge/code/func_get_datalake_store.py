from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def get_datalake_store(self):
    self.log('Get properties for datalake store {0}'.format(self.name))
    datalake_store_obj = None
    account_dict = None
    try:
        datalake_store_obj = self.datalake_store_client.accounts.get(self.resource_group, self.name)
    except ResourceNotFoundError:
        pass
    if datalake_store_obj:
        account_dict = self.account_obj_to_dict(datalake_store_obj)
    return account_dict