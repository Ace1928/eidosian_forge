from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
def change_password_rest(self, owner_uuid, username):
    body = {'password': self.parameters['set_password']}
    error = self.patch_account(owner_uuid, username, body)
    if error:
        if 'message' in error and self.is_repeated_password(error['message']):
            self.module.warn('Password was not changed: %s' % error['message'])
            return False
        self.module.fail_json(msg='Error while updating user password: %s' % error)
    return True