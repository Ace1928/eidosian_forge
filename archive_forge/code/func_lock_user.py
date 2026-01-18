from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
def lock_user(self):
    user = self.get_user()
    if not user:
        user = self.present_user()
    if user['state'].lower() == 'disabled':
        user = self.enable_user()
    if user['state'].lower() != 'locked':
        self.result['changed'] = True
        args = {'id': user['id']}
        if not self.module.check_mode:
            res = self.query_api('lockUser', **args)
            user = res['user']
    return user