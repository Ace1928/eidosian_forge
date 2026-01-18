from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import to_native
from ansible_collections.netapp.azure.plugins.module_utils.azure_rm_netapp_common import AzureRMNetAppModuleBase
from ansible_collections.netapp.azure.plugins.module_utils.netapp_module import NetAppModule
def modify_azure_netapp_capacity_pool(self, modify):
    """
            Modify a capacity pool for the given Azure NetApp Account
            :return: None
        """
    options = self.na_helper.get_not_none_values_from_dict(self.parameters, ['location', 'service_level', 'size', 'tags'])
    capacity_pool_body = CapacityPool(**options)
    try:
        response = self.get_method('pools', 'update')(body=capacity_pool_body, resource_group_name=self.parameters['resource_group'], account_name=self.parameters['account_name'], pool_name=self.parameters['name'])
        while response.done() is not True:
            response.result(10)
    except (CloudError, AzureError) as error:
        self.module.fail_json(msg='Error modifying capacity pool %s for Azure NetApp account %s: %s' % (self.parameters['name'], self.parameters['account_name'], to_native(error)), exception=traceback.format_exc())