from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
def lock_unlock_user_rest(self, owner_uuid, username, value=None):
    body = {'locked': value}
    error = self.patch_account(owner_uuid, username, body)
    if error:
        self.module.fail_json(msg='Error while locking/unlocking user: %s' % error)