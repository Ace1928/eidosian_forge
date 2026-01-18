from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def modify_key_manager_rest(self, modify, current=None, return_error=False):
    key_servers = self.na_helper.safe_get(modify, ['external', 'servers'])
    if key_servers:
        del modify['external']['servers']
        if not modify['external']:
            del modify['external']
    if modify:
        api = 'security/key-managers'
        body = self.create_body(modify)
        dummy, error = rest_generic.patch_async(self.rest_api, api, self.uuid, body)
        if error:
            if return_error:
                return error
            resource = 'cluster' if self.parameters.get('vserver') is None else self.parameters['vserver']
            self.module.fail_json(msg='Error modifying key manager for %s: %s' % (resource, error))
    if key_servers:
        self.update_key_server_list(current)
    return None