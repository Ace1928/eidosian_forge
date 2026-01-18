from __future__ import absolute_import, division, print_function
import copy
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.dnac.plugins.module_utils.dnac import (
def create_device_credentials(self):
    """
        Create Global Device Credential to the Cisco DNA
        Center based on the provided playbook details.
        Check the return value of the API with check_return_status().

        Parameters:
            self

        Returns:
            self
        """
    result_global_credential = self.result.get('response')[0].get('globalCredential')
    want_create = self.want.get('want_create')
    if not want_create:
        result_global_credential.update({'No Creation': {'response': 'No Response', 'msg': 'No Creation is available'}})
        return self
    credential_params = want_create
    self.log('Creating global credential API input parameters: {0}'.format(credential_params), 'DEBUG')
    response = self.dnac._exec(family='discovery', function='create_global_credentials_v2', params=credential_params)
    self.log("Received API response from 'create_global_credentials_v2': {0}".format(response), 'DEBUG')
    validation_string = 'global credential addition performed'
    self.check_task_response_status(response, validation_string).check_return_status()
    self.log('Global credential created successfully', 'INFO')
    result_global_credential.update({'Creation': {'response': credential_params, 'msg': 'Global Credential Created Successfully'}})
    self.msg = 'Global Device Credential Created Successfully'
    self.status = 'success'
    return self