from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import to_native
from ansible_collections.netapp.azure.plugins.module_utils.azure_rm_netapp_common import AzureRMNetAppModuleBase
from ansible_collections.netapp.azure.plugins.module_utils.netapp_module import NetAppModule
def create_azure_netapp_volume(self):
    """
            Create a volume for the given Azure NetApp Account
            :return: None
        """
    options = self.na_helper.get_not_none_values_from_dict(self.parameters, ['protocol_types', 'service_level', 'tags', 'usage_threshold'])
    rules = self.get_export_policy_rules()
    if rules is not None:
        options['export_policy'] = rules
    subnet_id = '/subscriptions/%s/resourceGroups/%s/providers/Microsoft.Network/virtualNetworks/%s/subnets/%s' % (self.azure_auth.subscription_id, self.parameters['resource_group'] if self.parameters.get('vnet_resource_group_for_subnet') is None else self.parameters['vnet_resource_group_for_subnet'], self.parameters['virtual_network'], self.parameters['subnet_name'])
    volume_body = Volume(location=self.parameters['location'], creation_token=self.parameters['file_path'], subnet_id=subnet_id, **options)
    try:
        result = self.get_method('volumes', 'create_or_update')(body=volume_body, resource_group_name=self.parameters['resource_group'], account_name=self.parameters['account_name'], pool_name=self.parameters['pool_name'], volume_name=self.parameters['name'])
        while result.done() is not True:
            result.result(10)
    except (CloudError, ValidationError, AzureError) as error:
        self.module.fail_json(msg='Error creating volume %s for Azure NetApp account %s and subnet ID %s: %s' % (self.parameters['name'], self.parameters['account_name'], subnet_id, to_native(error)), exception=traceback.format_exc())