from __future__ import absolute_import, division, print_function
import time
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase, format_resource_id
def get_natgateway(self):
    """
        Gets the properties of the specified NAT Gateway.

        :return: deserialized NAT Gateway instance state dictionary
        """
    self.log('Checking if the NAT Gateway instance {0} is present'.format(self.name))
    found = False
    try:
        response = self.network_client.nat_gateways.get(resource_group_name=self.resource_group, nat_gateway_name=self.name)
        found = True
        self.log('Response : {0}'.format(response))
        self.log('NAT Gateway instance : {0} found'.format(response.name))
    except ResourceNotFoundError as e:
        self.log('Did not find the NAT Gateway instance.')
    if found is True:
        return response.as_dict()
    return False