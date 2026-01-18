from __future__ import absolute_import, division, print_function
import time
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common_ext import AzureRMModuleBaseExt
def create_update_sqlserver(self):
    """
        Creates or updates SQL Server with the specified configuration.

        :return: deserialized SQL Server instance state dictionary
        """
    self.log('Creating / Updating the SQL Server instance {0}'.format(self.name))
    try:
        response = self.sql_client.servers.begin_create_or_update(resource_group_name=self.resource_group, server_name=self.name, parameters=self.parameters)
        if isinstance(response, LROPoller):
            response = self.get_poller_result(response)
    except Exception as exc:
        self.log('Error attempting to create the SQL Server instance.')
        self.fail('Error creating the SQL Server instance: {0}'.format(str(exc)))
    return response.as_dict()