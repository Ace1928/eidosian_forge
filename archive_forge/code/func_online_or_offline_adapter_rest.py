from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def online_or_offline_adapter_rest(self, status, adapter_name):
    api = 'network/fc/ports'
    body = {'enabled': True if status == 'up' else False}
    dummy, error = rest_generic.patch_async(self.rest_api, api, self.adapters_uuids[adapter_name], body)
    if error:
        self.module.fail_json(msg='Error trying to %s fc-adapter %s: %s' % (status, adapter_name, to_native(error)))