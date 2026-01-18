from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def create_role_rest(self):
    api = 'security/roles'
    body = {'name': self.parameters['name']}
    if self.parameters.get('vserver'):
        body['owner.name'] = self.parameters['vserver']
    body['privileges'] = self.parameters['privileges']
    dummy, error = rest_generic.post_async(self.rest_api, api, body, job_timeout=120)
    if error:
        self.module.fail_json(msg='Error creating role %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())