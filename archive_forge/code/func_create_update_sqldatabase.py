from __future__ import absolute_import, division, print_function
import time
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase, format_resource_id
def create_update_sqldatabase(self):
    """
        Creates or updates SQL Database with the specified configuration.

        :return: deserialized SQL Database instance state dictionary
        """
    self.log('Creating / Updating the SQL Database instance {0}'.format(self.name))
    try:
        response = self.sql_client.databases.begin_create_or_update(resource_group_name=self.resource_group, server_name=self.server_name, database_name=self.name, parameters=self.parameters)
        if isinstance(response, LROPoller):
            response = self.get_poller_result(response)
    except Exception as exc:
        self.log('Error attempting to create the SQL Database instance.')
        self.fail('Error creating the SQL Database instance: {0}'.format(str(exc)))
    return response.as_dict()