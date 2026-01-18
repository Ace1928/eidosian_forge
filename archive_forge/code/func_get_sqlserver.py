from __future__ import absolute_import, division, print_function
import time
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common_ext import AzureRMModuleBaseExt
def get_sqlserver(self):
    """
        Gets the properties of the specified SQL Server.

        :return: deserialized SQL Server instance state dictionary
        """
    self.log('Checking if the SQL Server instance {0} is present'.format(self.name))
    found = False
    try:
        response = self.sql_client.servers.get(resource_group_name=self.resource_group, server_name=self.name)
        found = True
        self.log('Response : {0}'.format(response))
        self.log('SQL Server instance : {0} found'.format(response.name))
    except ResourceNotFoundError:
        self.log('Did not find the SQL Server instance.')
    if found is True:
        return response.as_dict()
    return False