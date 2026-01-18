from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common_ext import AzureRMModuleBaseExt
def get_batchaccount(self):
    """
        Gets the properties of the specified Batch Account
        :return: deserialized Batch Account instance state dictionary
        """
    self.log('Checking if the Batch Account instance {0} is present'.format(self.name))
    try:
        response = self.mgmt_client.batch_account.get(resource_group_name=self.resource_group, account_name=self.name)
        self.log('Response : {0}'.format(response))
        self.log('Batch Account instance : {0} found'.format(response.name))
    except ResourceNotFoundError as e:
        self.log('Did not find the Batch Account instance.')
        return
    return response.as_dict()