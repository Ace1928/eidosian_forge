from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic, rest_vserver
def create_cifs_local_user(self):
    api = 'protocols/cifs/local-users'
    body = {'svm.uuid': self.svm_uuid, 'name': self.parameters['name']}
    if self.parameters.get('user_password') is not None:
        body['password'] = self.parameters['user_password']
    if self.parameters.get('full_name') is not None:
        body['full_name'] = self.parameters['full_name']
    if self.parameters.get('description') is not None:
        body['description'] = self.parameters['description']
    if self.parameters.get('account_disabled') is not None:
        body['account_disabled'] = self.parameters['account_disabled']
    dummy, error = rest_generic.post_async(self.rest_api, api, body)
    if error:
        self.module.fail_json(msg='Error creating CIFS local users with name %s: %s' % (self.parameters['name'], error))