from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
def get_network_acl_rule(self):
    args = {'aclid': self.get_network_acl(key='id'), 'account': self.get_account(key='name'), 'domainid': self.get_domain(key='id'), 'projectid': self.get_project(key='id')}
    network_acl_rules = self.query_api('listNetworkACLs', **args)
    for acl_rule in network_acl_rules.get('networkacl', []):
        if acl_rule['number'] == self.module.params.get('rule_position'):
            return acl_rule
    return None