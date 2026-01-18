from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.facts.facts import Facts
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.utils.utils import dict_to_set
def remove_command_from_config_list(self, vlan_id, cmd, commands):
    if vlan_id not in commands and cmd != 'vlan':
        commands.insert(0, vlan_id)
    elif cmd == 'vlan':
        commands.append('no %s' % vlan_id)
        return commands
    commands.append('no %s' % cmd)
    return commands