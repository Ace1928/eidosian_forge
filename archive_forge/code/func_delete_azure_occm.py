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
def delete_azure_occm(self):
    """
        Delete OCCM
        :return:
            None
        """
    try:
        compute_client = get_client_from_cli_profile(ComputeManagementClient)
        vm_delete = compute_client.virtual_machines.begin_delete(self.parameters['resource_group'], self.parameters['name'])
        while not vm_delete.done():
            vm_delete.wait(2)
    except CloudError as error:
        self.module.fail_json(msg=to_native(error), exception=traceback.format_exc())
    try:
        network_client = get_client_from_cli_profile(NetworkManagementClient)
        interface_delete = network_client.network_interfaces.begin_delete(self.parameters['resource_group'], self.parameters['name'] + '-nic')
        while not interface_delete.done():
            interface_delete.wait(2)
    except CloudError as error:
        self.module.fail_json(msg=to_native(error), exception=traceback.format_exc())
    try:
        storage_client = get_client_from_cli_profile(StorageManagementClient)
        storage_client.storage_accounts.delete(self.parameters['resource_group'], self.parameters['storage_account'])
    except CloudError as error:
        self.module.fail_json(msg=to_native(error), exception=traceback.format_exc())
    try:
        network_client = get_client_from_cli_profile(NetworkManagementClient)
        public_ip_addresses_delete = network_client.public_ip_addresses.begin_delete(self.parameters['resource_group'], self.parameters['name'] + '-ip')
        while not public_ip_addresses_delete.done():
            public_ip_addresses_delete.wait(2)
    except CloudError as error:
        self.module.fail_json(msg=to_native(error), exception=traceback.format_exc())
    try:
        resource_client = get_client_from_cli_profile(ResourceManagementClient)
        deployments_delete = resource_client.deployments.begin_delete(self.parameters['resource_group'], self.parameters['name'] + '-ip')
        while not deployments_delete.done():
            deployments_delete.wait(5)
    except CloudError as error:
        self.module.fail_json(msg=to_native(error), exception=traceback.format_exc())
    retries = 16
    while retries > 0:
        occm_resp, error = self.na_helper.check_occm_status(self.rest_api, self.parameters['client_id'])
        if error is not None:
            self.module.fail_json(msg='Error: Not able to get occm status: %s, %s' % (str(error), str(occm_resp)))
        if occm_resp['agent']['status'] != 'active':
            break
        else:
            time.sleep(10)
        retries -= 1
    if retries == 0:
        return self.module.fail_json(msg='Taking too long for instance to finish terminating')
    client = self.rest_api.format_client_id(self.parameters['client_id'])
    error = self.na_helper.delete_occm_agents(self.rest_api, [{'agentId': client}])
    if error:
        self.module.fail_json(msg='Error: unexpected response on deleting OCCM: %s' % str(error))