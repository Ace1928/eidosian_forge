from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (
def reset_password(self):
    args = {'id': self.get_vm(key='id')}
    res = None
    self.result['changed'] = True
    if not self.module.check_mode:
        res = self.query_api('resetPasswordForVirtualMachine', **args)
        poll_async = self.module.params.get('poll_async')
        if res and poll_async:
            res = self.poll_job(res, 'virtualmachine')
    if res and 'password' in res:
        self.password = res['password']
    return self.password