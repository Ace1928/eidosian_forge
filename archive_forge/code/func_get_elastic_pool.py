from __future__ import absolute_import, division, print_function
import time
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def get_elastic_pool(self):
    """
        Gets the properties of the specified SQL Elastic Pool.

        :return: deserialized SQL Elastic Pool instance state dictionary
        """
    found = False
    try:
        response = self.sql_client.elastic_pools.get(resource_group_name=self.resource_group, server_name=self.server_name, elastic_pool_name=self.name)
        found = True
    except ResourceNotFoundError:
        self.log('Did not find the SQL Elastic Pool instance.')
    if found is True:
        return self.format_item(response)
    return False