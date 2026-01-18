from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic, rest_volume
def flexcache_rest_modify(self, uuid):
    """ use PATCH to start prepopulating a FlexCache """
    mappings = dict(prepopulate='prepopulate')
    body = self.flexcache_rest_create_body(mappings)
    api = 'storage/flexcache/flexcaches'
    response, error = rest_generic.patch_async(self.rest_api, api, uuid, body, job_timeout=self.parameters['time_out'])
    self.na_helper.fail_on_error(error)
    return response