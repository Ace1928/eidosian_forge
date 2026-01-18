from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
def present_portforwarding_rule(self):
    portforwarding_rule = self.get_portforwarding_rule()
    if portforwarding_rule:
        portforwarding_rule = self.update_portforwarding_rule(portforwarding_rule)
    else:
        portforwarding_rule = self.create_portforwarding_rule()
    if portforwarding_rule:
        portforwarding_rule = self.ensure_tags(resource=portforwarding_rule, resource_type='PortForwardingRule')
        self.portforwarding_rule = portforwarding_rule
    return portforwarding_rule