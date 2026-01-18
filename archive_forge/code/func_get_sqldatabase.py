from __future__ import absolute_import, division, print_function
import time
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase, format_resource_id
def get_sqldatabase(self):
    """
        Gets the properties of the specified SQL Database.

        :return: deserialized SQL Database instance state dictionary
        """
    self.log('Checking if the SQL Database instance {0} is present'.format(self.name))
    found = False
    try:
        response = self.sql_client.databases.get(resource_group_name=self.resource_group, server_name=self.server_name, database_name=self.name)
        found = True
        self.log('Response : {0}'.format(response))
        self.log('SQL Database instance : {0} found'.format(response.name))
    except ResourceNotFoundError:
        self.log('Did not find the SQL Database instance.')
    if found is True:
        return response.as_dict()
    return False