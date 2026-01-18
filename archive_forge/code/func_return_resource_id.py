from __future__ import absolute_import, division, print_function
import time
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase, format_resource_id
def return_resource_id(self, resource):
    """
        Build an IP Address resource id from different inputs

        :return string containing the Azure id of the resource
        """
    if is_valid_resource_id(resource):
        return resource
    resource_dict = self.parse_resource_to_dict(resource)
    return format_resource_id(val=resource_dict['name'], subscription_id=resource_dict.get('subscription_id'), namespace='Microsoft.Network', types='publicIPAddresses', resource_group=resource_dict.get('resource_group'))