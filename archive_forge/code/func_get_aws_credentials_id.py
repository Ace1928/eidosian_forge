from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
import ansible_collections.netapp.cloudmanager.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.cloudmanager.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.cloudmanager.plugins.module_utils.netapp import CloudManagerRestAPI
def get_aws_credentials_id(self):
    """
        Get aws_credentials_id
        :return: AWS Credentials ID
        """
    api = '/fsx-ontap/aws-credentials/'
    api += self.parameters['tenant_id']
    response, error, dummy = self.rest_api.get(api, None, header=self.headers)
    if error:
        return (response, 'Error: getting aws_credentials_id %s' % error)
    for each in response:
        if each['name'] == self.parameters['aws_credentials_name']:
            return (each['id'], None)
    return (None, 'Error: aws_credentials_name not found')