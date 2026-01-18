from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
from ansible.module_utils.common.dict_transformations import _snake_to_camel
def delete_databaseaccount(self):
    """
        Deletes specified Database Account instance in the specified subscription and resource group.

        :return: True
        """
    self.log('Deleting the Database Account instance {0}'.format(self.name))
    try:
        response = self.mgmt_client.database_accounts.begin_delete(resource_group_name=self.resource_group, account_name=self.name)
        if isinstance(response, LROPoller):
            response = self.get_poller_result(response)
    except Exception as e:
        self.log('Error attempting to delete the Database Account instance.')
        self.fail('Error deleting the Database Account instance: {0}'.format(str(e)))
    return True