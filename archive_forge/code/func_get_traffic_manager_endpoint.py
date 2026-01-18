from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
from ansible.module_utils.common.dict_transformations import _snake_to_camel
def get_traffic_manager_endpoint(self):
    """
        Gets the properties of the specified Traffic Manager endpoint

        :return: deserialized Traffic Manager endpoint dict
        """
    self.log('Checking if Traffic Manager endpoint {0} is present'.format(self.name))
    try:
        response = self.traffic_manager_management_client.endpoints.get(self.resource_group, self.profile_name, self.type, self.name)
        self.log('Response : {0}'.format(response))
        return traffic_manager_endpoint_to_dict(response)
    except ResourceNotFoundError:
        self.log('Did not find the Traffic Manager endpoint.')
        return False