from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def get_role_rest(self):
    api = 'security/roles'
    if self.rest_api.meets_rest_minimum_version(self.use_rest, 9, 11, 1):
        fields = 'name,owner,privileges.path,privileges.access,privileges.query'
    else:
        fields = 'name,owner,privileges.path,privileges.access'
    params = {'name': self.parameters['name'], 'fields': fields}
    if self.parameters.get('vserver'):
        params['owner.name'] = self.parameters['vserver']
    else:
        params['scope'] = 'cluster'
    record, error = rest_generic.get_one_record(self.rest_api, api, params)
    if error:
        self.module.fail_json(msg='Error getting role %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())
    return self.format_record(record)