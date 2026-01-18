from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def list_app_settings(self):
    """
        List application settings
        :return: deserialized list response
        """
    self.log('List application setting')
    try:
        response = self.web_client.web_apps.list_application_settings(resource_group_name=self.resource_group, name=self.name)
        self.log('Response : {0}'.format(response))
        return response.properties
    except Exception as ex:
        self.fail('Failed to list application settings for web app {0} in resource group {1}: {2}'.format(self.name, self.resource_group, str(ex)))