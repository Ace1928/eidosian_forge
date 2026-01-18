from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.facts.facts import Facts
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.utils.utils import (
def del_commands(self, obj):
    commands = []
    if not obj or len(obj.keys()) == 1:
        return commands
    commands.append('interface ' + obj['name'])
    if 'transmit' in obj:
        commands.append('lldp transmit')
    if 'receive' in obj:
        commands.append('lldp receive')
    if 'management_address' in obj:
        commands.append('no lldp tlv-set management-address ' + obj['management_address'])
    if 'vlan' in obj:
        commands.append('no lldp tlv-set vlan ' + str(obj['vlan']))
    return commands