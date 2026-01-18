from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
def remove_firewall_rule(self):
    firewall_rule = self.get_firewall_rule()
    if firewall_rule:
        self.result['changed'] = True
        args = {'id': firewall_rule['id']}
        fw_type = self.module.params.get('type')
        if not self.module.check_mode:
            if fw_type == 'egress':
                res = self.query_api('deleteEgressFirewallRule', **args)
            else:
                res = self.query_api('deleteFirewallRule', **args)
            poll_async = self.module.params.get('poll_async')
            if poll_async:
                self.poll_job(res, 'firewallrule')
    return firewall_rule