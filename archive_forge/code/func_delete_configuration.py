from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def delete_configuration(self):
    self.log('Deleting the Configuration instance {0}'.format(self.name))
    try:
        response = self.mariadb_client.configurations.begin_create_or_update(resource_group_name=self.resource_group, server_name=self.server_name, configuration_name=self.name, parameters={'source': 'system-default'})
    except Exception as e:
        self.log('Error attempting to delete the Configuration instance.')
        self.fail('Error deleting the Configuration instance: {0}'.format(str(e)))
    return True