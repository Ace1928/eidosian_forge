from __future__ import absolute_import, division, print_function
import time
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase, format_resource_id
def create_update_natgateway(self):
    """
        Creates or updates NAT Gateway with the specified configuration.

        :return: deserialized NAT Gateway instance state dictionary
        """
    self.log('Creating / Updating the NAT Gateway instance {0}'.format(self.name))
    try:
        response = self.network_client.nat_gateways.begin_create_or_update(resource_group_name=self.resource_group, nat_gateway_name=self.name, parameters=self.parameters)
        if isinstance(response, LROPoller):
            response = self.get_poller_result(response)
    except Exception as exc:
        self.log('Error attempting to create the NAT Gateway instance.')
        self.fail('Error creating the NAT Gateway instance: {0}'.format(str(exc)))
    return response.as_dict()