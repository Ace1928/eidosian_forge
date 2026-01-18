from __future__ import absolute_import, division, print_function
import traceback
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def create_efficiency_policy_rest(self):
    api = 'storage/volume-efficiency-policies'
    body = {'svm.name': self.parameters['vserver'], 'name': self.parameters['policy_name']}
    create_or_modify_body = self.form_create_or_modify_body(self.parameters)
    if create_or_modify_body:
        body.update(create_or_modify_body)
    dummy, error = rest_generic.post_async(self.rest_api, api, body)
    if error:
        self.module.fail_json(msg='Error creating efficiency policy %s: %s' % (self.parameters['policy_name'], error))