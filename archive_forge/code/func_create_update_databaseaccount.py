from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
from ansible.module_utils.common.dict_transformations import _snake_to_camel
def create_update_databaseaccount(self):
    """
        Creates or updates Database Account with the specified configuration.

        :return: deserialized Database Account instance state dictionary
        """
    self.log('Creating / Updating the Database Account instance {0}'.format(self.name))
    try:
        response = self.mgmt_client.database_accounts.begin_create_or_update(resource_group_name=self.resource_group, account_name=self.name, create_update_parameters=self.parameters)
        if isinstance(response, LROPoller):
            response = self.get_poller_result(response)
    except Exception as exc:
        self.log('Error attempting to create the Database Account instance.')
        self.fail('Error creating the Database Account instance: {0}'.format(str(exc)))
    return response.as_dict()