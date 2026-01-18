from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic, rest_vserver
def modify_cifs_local_user(self, modify):
    api = 'protocols/cifs/local-users'
    uuids = '%s/%s' % (self.svm_uuid, self.sid)
    body = {}
    if modify.get('full_name') is not None:
        body['full_name'] = self.parameters['full_name']
    if modify.get('description') is not None:
        body['description'] = self.parameters['description']
    if modify.get('account_disabled') is not None:
        body['account_disabled'] = self.parameters['account_disabled']
    if self.parameters['set_password'] and modify.get('user_password') is not None:
        body['password'] = self.parameters['user_password']
    dummy, error = rest_generic.patch_async(self.rest_api, api, uuids, body)
    if error:
        self.module.fail_json(msg='Error while modifying CIFS local user: %s' % error)