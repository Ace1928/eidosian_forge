from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.facts.facts import Facts
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.utils.utils import (
def del_attribs(self, obj):
    commands = []
    if not obj or len(obj.keys()) == 1:
        return commands
    commands.append('interface ' + obj['name'])
    if 'graceful' in obj:
        commands.append('lacp graceful-convergence')
    if 'vpc' in obj:
        commands.append('no lacp vpn-convergence')
    if 'suspend_individual' in obj:
        commands.append('lacp suspend_individual')
    if 'mode' in obj:
        commands.append('no lacp mode ' + obj['mode'])
    if 'max' in obj:
        commands.append('no lacp max-bundle')
    if 'min' in obj:
        commands.append('no lacp min-links')
    if 'port_priority' in obj:
        commands.append('no lacp port-priority')
    if 'rate' in obj:
        commands.append('no lacp rate')
    return commands