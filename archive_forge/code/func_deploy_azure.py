from __future__ import absolute_import, division, print_function
import traceback
import time
import base64
import json
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.cloudmanager.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.cloudmanager.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.cloudmanager.plugins.module_utils.netapp import CloudManagerRestAPI
def deploy_azure(self):
    """
        Create Cloud Manager connector for Azure
        :return: client_id
        """
    user_data, client_id = self.register_agent_to_service()
    template = json.loads(self.na_helper.call_template())
    params = json.loads(self.na_helper.call_parameters())
    params['adminUsername']['value'] = self.parameters['admin_username']
    params['adminPassword']['value'] = self.parameters['admin_password']
    params['customData']['value'] = json.dumps(user_data)
    params['location']['value'] = self.parameters['location']
    params['virtualMachineName']['value'] = self.parameters['name']
    params['storageAccount']['value'] = self.parameters['storage_account']
    if self.rest_api.environment == 'stage':
        params['environment']['value'] = self.rest_api.environment
    if '/subscriptions' in self.parameters['vnet_name']:
        network = self.parameters['vnet_name']
    elif self.parameters.get('vnet_resource_group') is not None:
        network = '/subscriptions/%s/resourceGroups/%s/providers/Microsoft.Network/virtualNetworks/%s' % (self.parameters['subscription_id'], self.parameters['vnet_resource_group'], self.parameters['vnet_name'])
    else:
        network = '/subscriptions/%s/resourceGroups/%s/providers/Microsoft.Network/virtualNetworks/%s' % (self.parameters['subscription_id'], self.parameters['resource_group'], self.parameters['vnet_name'])
    if '/subscriptions' in self.parameters['subnet_name']:
        subnet = self.parameters['subnet_name']
    else:
        subnet = '%s/subnets/%s' % (network, self.parameters['subnet_name'])
    if self.parameters.get('network_security_resource_group') is not None:
        network_security_group_name = '/subscriptions/%s/resourceGroups/%s/providers/Microsoft.Network/networkSecurityGroups/%s' % (self.parameters['subscription_id'], self.parameters['network_security_resource_group'], self.parameters['network_security_group_name'])
    else:
        network_security_group_name = '/subscriptions/%s/resourceGroups/%s/providers/Microsoft.Network/networkSecurityGroups/%s' % (self.parameters['subscription_id'], self.parameters['resource_group'], self.parameters['network_security_group_name'])
    params['virtualNetworkId']['value'] = network
    params['networkSecurityGroupName']['value'] = network_security_group_name
    params['virtualMachineSize']['value'] = self.parameters['virtual_machine_size']
    params['subnetId']['value'] = subnet
    try:
        resource_client = get_client_from_cli_profile(ResourceManagementClient)
        resource_client.resource_groups.create_or_update(self.parameters['resource_group'], {'location': self.parameters['location']})
        deployment_properties = {'mode': 'Incremental', 'template': template, 'parameters': params}
        resource_client.deployments.begin_create_or_update(self.parameters['resource_group'], self.parameters['name'], Deployment(properties=deployment_properties))
    except CloudError as error:
        self.module.fail_json(msg='Error in deploy_azure: %s' % to_native(error), exception=traceback.format_exc())
    time.sleep(120)
    retries = 30
    while retries > 0:
        occm_resp, error = self.na_helper.check_occm_status(self.rest_api, client_id)
        if error is not None:
            self.module.fail_json(msg='Error: Not able to get occm status: %s, %s' % (str(error), str(occm_resp)))
        if occm_resp['agent']['status'] == 'active':
            break
        else:
            time.sleep(30)
        retries -= 1
    if retries == 0:
        return self.module.fail_json(msg='Taking too long for OCCM agent to be active or not properly setup')
    try:
        compute_client = get_client_from_cli_profile(ComputeManagementClient)
        vm = compute_client.virtual_machines.get(self.parameters['resource_group'], self.parameters['name'])
    except CloudError as error:
        return self.module.fail_json(msg='Error in deploy_azure (get identity): %s' % to_native(error), exception=traceback.format_exc())
    principal_id = vm.identity.principal_id
    return (client_id, principal_id)