from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
def remove_nic(self, nic):
    self.result['changed'] = True
    args = {'virtualmachineid': self.get_vm(key='id'), 'nicid': nic['id']}
    if not self.module.check_mode:
        res = self.query_api('removeNicFromVirtualMachine', **args)
        if self.module.params.get('poll_async'):
            self.poll_job(res, 'virtualmachine')
    return nic