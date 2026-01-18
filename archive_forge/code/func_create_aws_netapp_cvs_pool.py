from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import ansible_collections.netapp.aws.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.aws.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.aws.plugins.module_utils.netapp import AwsCvsRestAPI
def create_aws_netapp_cvs_pool(self):
    """
        Create a pool
        """
    api = 'Pools'
    for key in ['serviceLevel', 'sizeInBytes', 'vendorID']:
        if key not in self.parameters.keys() or self.parameters[key] is None:
            self.module.fail_json(changed=False, msg="Mandatory key '%s' required" % key)
    pool = {'name': self.parameters['name'], 'region': self.parameters['region'], 'serviceLevel': self.parameters['serviceLevel'], 'sizeInBytes': self.parameters['sizeInBytes'], 'vendorID': self.parameters['vendorID']}
    dummy, error = self.rest_api.post(api, pool)
    if error is not None:
        self.module.fail_json(changed=False, msg=error)