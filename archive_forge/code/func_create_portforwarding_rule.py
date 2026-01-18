from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
def create_portforwarding_rule(self):
    args = {'protocol': self.module.params.get('protocol'), 'publicport': self.module.params.get('public_port'), 'publicendport': self.get_or_fallback('public_end_port', 'public_port'), 'privateport': self.module.params.get('private_port'), 'privateendport': self.get_or_fallback('private_end_port', 'private_port'), 'openfirewall': self.module.params.get('open_firewall'), 'vmguestip': self.get_vm_guest_ip(), 'ipaddressid': self.get_ip_address(key='id'), 'virtualmachineid': self.get_vm(key='id'), 'account': self.get_account(key='name'), 'domainid': self.get_domain(key='id'), 'networkid': self.get_network(key='id')}
    portforwarding_rule = None
    self.result['changed'] = True
    if not self.module.check_mode:
        portforwarding_rule = self.query_api('createPortForwardingRule', **args)
        poll_async = self.module.params.get('poll_async')
        if poll_async:
            portforwarding_rule = self.poll_job(portforwarding_rule, 'portforwardingrule')
    return portforwarding_rule