from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.facts.facts import Facts
def process_nested_dict(self, val):
    nested_commands = []
    for k, v in val.items():
        if isinstance(v, dict):
            for k1, v1 in v.items():
                com1 = 'lldp tlv-select '
                com2 = ''
                if 'system' in k:
                    com2 = 'system-' + k1
                elif 'management_address' in k:
                    com2 = 'management-address ' + k1
                elif 'port' in k:
                    com2 = 'port-' + k1
                com1 += com2
                com1 = self.negate_command(com1, v1)
                nested_commands.append(com1)
        else:
            com1 = 'lldp tlv-select '
            if 'power_management' in k:
                com1 += 'power-management'
            else:
                com1 += k
            com1 = self.negate_command(com1, v)
            nested_commands.append(com1)
    return nested_commands