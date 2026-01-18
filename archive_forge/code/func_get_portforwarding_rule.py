from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
def get_portforwarding_rule(self):
    if not self.portforwarding_rule:
        protocol = self.module.params.get('protocol')
        public_port = self.module.params.get('public_port')
        args = {'ipaddressid': self.get_ip_address(key='id'), 'account': self.get_account(key='name'), 'domainid': self.get_domain(key='id'), 'projectid': self.get_project(key='id')}
        portforwarding_rules = self.query_api('listPortForwardingRules', **args)
        if portforwarding_rules and 'portforwardingrule' in portforwarding_rules:
            for rule in portforwarding_rules['portforwardingrule']:
                if protocol == rule['protocol'] and public_port == int(rule['publicport']):
                    self.portforwarding_rule = rule
                    break
    return self.portforwarding_rule