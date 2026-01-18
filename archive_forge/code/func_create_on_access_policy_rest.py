from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic, rest_vserver
def create_on_access_policy_rest(self):
    api = 'protocols/vscan/%s/on-access-policies' % self.svm_uuid
    body = {'name': self.parameters['policy_name']}
    body.update(self.form_create_or_modify_body(self.parameters))
    dummy, error = rest_generic.post_async(self.rest_api, api, body)
    if error:
        self.module.fail_json(msg='Error creating Vscan on Access Policy %s: %s' % (self.parameters['policy_name'], to_native(error)))