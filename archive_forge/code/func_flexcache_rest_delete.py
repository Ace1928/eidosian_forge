from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic, rest_volume
def flexcache_rest_delete(self, current):
    """
        Delete the flexcache using REST DELETE method.
        """
    response = None
    uuid = current.get('uuid')
    if uuid is None:
        error = 'Error, no uuid in current: %s' % str(current)
        self.na_helper.fail_on_error(error)
    api = 'storage/flexcache/flexcaches'
    rto = netapp_utils.get_feature(self.module, 'flexcache_delete_return_timeout')
    response, error = rest_generic.delete_async(self.rest_api, api, uuid, timeout=rto, job_timeout=self.parameters['time_out'])
    self.na_helper.fail_on_error(error)
    return response