from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def create_update_configuration(self):
    self.log('Creating / Updating the Configuration instance {0}'.format(self.name))
    try:
        response = self.mariadb_client.configurations.begin_create_or_update(resource_group_name=self.resource_group, server_name=self.server_name, configuration_name=self.name, parameters={'value': self.value, 'source': 'user-override'})
        if isinstance(response, LROPoller):
            response = self.get_poller_result(response)
    except Exception as exc:
        self.log('Error attempting to create the Configuration instance.')
        self.fail('Error creating the Configuration instance: {0}'.format(str(exc)))
    return response.as_dict()