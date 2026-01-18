from __future__ import absolute_import, division, print_function
import copy
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.dnac.plugins.module_utils.dnac import (
def delete_device_credential(self, config):
    """
        Delete Global Device Credential in Cisco DNA Center based on the playbook details.
        Check the return value of the API with check_return_status().

        Parameters:
            config (dict) - Playbook details containing Global Device Credential information.
            self - The current object details.

        Returns:
            self
        """
    result_global_credential = self.result.get('response')[0].get('globalCredential')
    have_values = self.have.get('globalCredential')
    final_response = {}
    self.log('Global device credentials to be deleted: {0}'.format(have_values), 'DEBUG')
    credential_mapping = {'cliCredential': 'cli_credential', 'snmpV2cRead': 'snmp_v2c_read', 'snmpV2cWrite': 'snmp_v2c_write', 'snmpV3': 'snmp_v3', 'httpsRead': 'https_read', 'httpsWrite': 'https_write'}
    for item in have_values:
        config_itr = 0
        final_response.update({item: []})
        for value in have_values.get(item):
            if value is None:
                self.log('Credential Name: {0}'.format(item), 'DEBUG')
                self.log('Credential Item: {0}'.format(config.get('global_credential_details').get(credential_mapping.get(item))), 'DEBUG')
                final_response.get(item).append(str(config.get('global_credential_details').get(credential_mapping.get(item))[config_itr]) + ' is not found.')
                continue
            _id = have_values.get(item)[config_itr].get('id')
            response = self.dnac._exec(family='discovery', function='delete_global_credential_v2', params={'id': _id})
            self.log("Received API response for 'delete_global_credential_v2': {0}".format(response), 'DEBUG')
            validation_string = 'global credential deleted successfully'
            self.check_task_response_status(response, validation_string).check_return_status()
            final_response.get(item).append(_id)
            config_itr = config_itr + 1
    self.log('Deleting device credential API input parameters: {0}'.format(final_response), 'DEBUG')
    self.log('Successfully deleted global device credential.', 'INFO')
    result_global_credential.update({'Deletion': {'response': final_response, 'msg': 'Global Device Credentials Deleted Successfully'}})
    self.msg = 'Global Device Credentials Updated Successfully'
    self.status = 'success'
    return self