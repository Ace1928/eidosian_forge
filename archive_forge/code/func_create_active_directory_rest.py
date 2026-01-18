from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def create_active_directory_rest(self):
    api = 'protocols/active-directory'
    body = {'svm.name': self.parameters['vserver'], 'name': self.parameters['account_name'], 'username': self.parameters['admin_username'], 'password': self.parameters['admin_password']}
    if self.parameters.get('domain'):
        body['fqdn'] = self.parameters['domain']
    if self.parameters.get('force_account_overwrite'):
        body['force_account_overwrite'] = self.parameters['force_account_overwrite']
    if self.parameters.get('organizational_unit'):
        body['organizational_unit'] = self.parameters['organizational_unit']
    dummy, error = rest_generic.post_async(self.rest_api, api, body)
    if error:
        self.module.fail_json(msg='Error creating vserver Active Directory %s: %s' % (self.parameters['account_name'], to_native(error)), exception=traceback.format_exc())