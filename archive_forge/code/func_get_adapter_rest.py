from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def get_adapter_rest(self):
    api = 'private/cli/ucadmin'
    params = {'node': self.parameters['node_name'], 'adapter': self.parameters['adapter_name'], 'fields': 'pending_mode,pending_type,current_mode,current_type,status_admin'}
    record, error = rest_generic.get_one_record(self.rest_api, api, params)
    if error:
        self.module.fail_json(msg='Error fetching ucadapter details: %s: %s' % (self.parameters['node_name'], to_native(error)))
    if record:
        return {'mode': self.na_helper.safe_get(record, ['current_mode']), 'pending-mode': self.na_helper.safe_get(record, ['pending_mode']), 'type': self.na_helper.safe_get(record, ['current_type']), 'pending-type': self.na_helper.safe_get(record, ['pending_type']), 'status': self.na_helper.safe_get(record, ['status_admin'])}
    return None