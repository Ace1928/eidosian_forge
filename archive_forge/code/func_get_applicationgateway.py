from __future__ import absolute_import, division, print_function
import time
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
from copy import deepcopy
from ansible.module_utils.common.dict_transformations import (
def get_applicationgateway(self):
    """
        Gets the properties of the specified Application Gateway.

        :return: deserialized Application Gateway instance state dictionary
        """
    self.log('Checking if the Application Gateway instance {0} is present'.format(self.name))
    found = False
    try:
        response = self.network_client.application_gateways.get(resource_group_name=self.resource_group, application_gateway_name=self.name)
        found = True
        self.log('Response : {0}'.format(response))
        self.log('Application Gateway instance : {0} found'.format(response.name))
    except ResourceNotFoundError as e:
        self.log('Did not find the Application Gateway instance.')
    if found is True:
        return response.as_dict()
    return False