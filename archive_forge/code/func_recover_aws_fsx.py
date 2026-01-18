from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
import ansible_collections.netapp.cloudmanager.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.cloudmanager.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.cloudmanager.plugins.module_utils.netapp import CloudManagerRestAPI
def recover_aws_fsx(self):
    """
        recover aws_fsx
        """
    json = {'name': self.parameters['name'], 'region': self.parameters['region'], 'workspaceId': self.parameters['workspace_id'], 'credentialsId': self.aws_credentials_id, 'fileSystemId': self.parameters['file_system_id']}
    api_url = '/fsx-ontap/working-environments/%s/recover' % self.parameters['tenant_id']
    response, error, dummy = self.rest_api.post(api_url, json, header=self.headers)
    if error is not None:
        self.module.fail_json(msg='Error: unexpected response on recovering AWS FSx: %s, %s' % (error, response))