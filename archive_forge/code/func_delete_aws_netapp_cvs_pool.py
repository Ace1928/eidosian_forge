from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import ansible_collections.netapp.aws.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.aws.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.aws.plugins.module_utils.netapp import AwsCvsRestAPI
def delete_aws_netapp_cvs_pool(self, pool_id):
    """
        Delete a pool
        """
    api = 'Pools/' + pool_id
    data = None
    dummy, error = self.rest_api.delete(api, data)
    if error is not None:
        self.module.fail_json(changed=False, msg=error)