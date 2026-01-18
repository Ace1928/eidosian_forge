from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def get_webapp_configuration(self):
    """
        Get  web app configuration
        :return: deserialized  web app configuration response
        """
    self.log('Get web app configuration')
    try:
        response = self.web_client.web_apps.get_configuration(resource_group_name=self.resource_group, name=self.name)
        self.log('Response : {0}'.format(response))
        return response
    except ResourceNotFoundError as ex:
        self.log('Failed to get configuration for web app {0} in resource group {1}: {2}'.format(self.name, self.resource_group, str(ex)))
        return False