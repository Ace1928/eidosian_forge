from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (
def create_static_nat(self, ip_address):
    self.result['changed'] = True
    args = {'virtualmachineid': self.get_vm(key='id'), 'ipaddressid': ip_address['id'], 'vmguestip': self.get_vm_guest_ip(), 'networkid': self.get_network(key='id')}
    if not self.module.check_mode:
        self.query_api('enableStaticNat', **args)
        self.ip_address = None
        ip_address = self.get_ip_address()
    return ip_address