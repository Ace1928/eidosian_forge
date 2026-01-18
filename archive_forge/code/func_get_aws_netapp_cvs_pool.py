from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import ansible_collections.netapp.aws.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.aws.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.aws.plugins.module_utils.netapp import AwsCvsRestAPI
def get_aws_netapp_cvs_pool(self, name=None):
    """
        Returns Pool object if exists else Return None
        """
    pool_info = None
    if name is None:
        name = self.parameters['name']
    pools, error = self.rest_api.get('Pools')
    if error is None and pools is not None:
        for pool in pools:
            if 'name' in pool and pool['region'] == self.parameters['region']:
                if pool['name'] == name:
                    pool_info = pool
                    break
    return pool_info