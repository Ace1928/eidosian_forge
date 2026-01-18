from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import to_native
from ansible_collections.netapp.azure.plugins.module_utils.azure_rm_netapp_common import AzureRMNetAppModuleBase
from ansible_collections.netapp.azure.plugins.module_utils.netapp_module import NetAppModule
def get_azure_netapp_capacity_pool(self):
    """
            Returns capacity pool object for an existing pool
            Return None if capacity pool does not exist
        """
    try:
        capacity_pool_get = self.netapp_client.pools.get(self.parameters['resource_group'], self.parameters['account_name'], self.parameters['name'])
    except (CloudError, ResourceNotFoundError):
        return None
    return capacity_pool_get