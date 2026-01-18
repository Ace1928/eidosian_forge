from __future__ import absolute_import, division, print_function
import time
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common_ext import AzureRMModuleBaseExt
def delete_sqlserver(self):
    """
        Deletes specified SQL Server instance in the specified subscription and resource group.

        :return: True
        """
    self.log('Deleting the SQL Server instance {0}'.format(self.name))
    try:
        response = self.sql_client.servers.begin_delete(resource_group_name=self.resource_group, server_name=self.name)
        if isinstance(response, LROPoller):
            response = self.get_poller_result(response)
    except Exception as e:
        self.log('Error attempting to delete the SQL Server instance.')
        self.fail('Error deleting the SQL Server instance: {0}'.format(str(e)))
    return True