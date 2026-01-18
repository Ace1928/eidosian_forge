from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
def present_network_acl_rule(self):
    network_acl_rule = self.get_network_acl_rule()
    protocol = self.module.params.get('protocol')
    start_port = self.module.params.get('start_port')
    end_port = self.get_or_fallback('end_port', 'start_port')
    icmp_type = self.module.params.get('icmp_type')
    icmp_code = self.module.params.get('icmp_code')
    if protocol in ['tcp', 'udp'] and (start_port is None or end_port is None):
        self.module.fail_json(msg='protocol is %s but the following are missing: start_port, end_port' % protocol)
    elif protocol == 'icmp' and (icmp_type is None or icmp_code is None):
        self.module.fail_json(msg='protocol is icmp but the following are missing: icmp_type, icmp_code')
    elif protocol == 'by_number' and self.module.params.get('protocol_number') is None:
        self.module.fail_json(msg='protocol is by_number but the following are missing: protocol_number')
    if not network_acl_rule:
        network_acl_rule = self._create_network_acl_rule(network_acl_rule)
    else:
        network_acl_rule = self._update_network_acl_rule(network_acl_rule)
    if network_acl_rule:
        network_acl_rule = self.ensure_tags(resource=network_acl_rule, resource_type='NetworkACL')
    return network_acl_rule