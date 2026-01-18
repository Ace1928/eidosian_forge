from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import to_native
from ansible_collections.netapp.azure.plugins.module_utils.azure_rm_netapp_common import AzureRMNetAppModuleBase
from ansible_collections.netapp.azure.plugins.module_utils.netapp_module import NetAppModule
def modify_azure_netapp_volume(self):
    """
            Modify a volume for the given Azure NetApp Account
            :return: None
        """
    options = self.na_helper.get_not_none_values_from_dict(self.parameters, ['tags', 'usage_threshold'])
    volume_body = VolumePatch(**options)
    try:
        result = self.get_method('volumes', 'update')(body=volume_body, resource_group_name=self.parameters['resource_group'], account_name=self.parameters['account_name'], pool_name=self.parameters['pool_name'], volume_name=self.parameters['name'])
        while result.done() is not True:
            result.result(10)
    except (CloudError, ValidationError, AzureError) as error:
        self.module.fail_json(msg='Error modifying volume %s for Azure NetApp account %s: %s' % (self.parameters['name'], self.parameters['account_name'], to_native(error)), exception=traceback.format_exc())