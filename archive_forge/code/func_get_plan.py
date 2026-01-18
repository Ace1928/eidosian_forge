from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def get_plan(self):
    """
        Gets app service plan
        :return: deserialized app service plan dictionary
        """
    self.log('Get App Service Plan {0}'.format(self.name))
    try:
        response = self.web_client.app_service_plans.get(resource_group_name=self.resource_group, name=self.name)
        if response:
            self.log('Response : {0}'.format(response))
            self.log('App Service Plan : {0} found'.format(response.name))
            return appserviceplan_to_dict(response)
    except ResourceNotFoundError:
        self.log("Didn't find app service plan {0} in resource group {1}".format(self.name, self.resource_group))
    return False