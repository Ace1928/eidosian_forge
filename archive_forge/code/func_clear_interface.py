from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.arista.eos.plugins.module_utils.network.eos.facts.facts import Facts
from ansible_collections.arista.eos.plugins.module_utils.network.eos.utils.utils import (
def clear_interface(want, have):
    commands = []
    if 'mode' in have and want.get('mode') is None:
        commands.append('no switchport mode')
    if 'access' in have and (not want.get('access')):
        commands.append('no switchport access vlan')
    has_trunk = have.get('trunk') or {}
    wants_trunk = want.get('trunk') or {}
    if 'trunk_allowed_vlans' in has_trunk and 'trunk_allowed_vlans' not in wants_trunk:
        commands.append('no switchport trunk allowed vlan')
    if 'trunk_allowed_vlans' in has_trunk and 'trunk_allowed_vlans' in wants_trunk:
        for con in [want, have]:
            expand_trunk_allowed_vlans(con)
        want_allowed_vlans = want['trunk'].get('trunk_allowed_vlans')
        has_allowed_vlans = has_trunk.get('trunk_allowed_vlans')
        allowed_vlans = list(set(has_allowed_vlans.split(',')) - set(want_allowed_vlans.split(',')))
        if allowed_vlans:
            allowed_vlans = ','.join(['{0}'.format(vlan) for vlan in allowed_vlans])
            commands.append('switchport trunk allowed vlan remove {0}'.format(allowed_vlans))
    if 'native_vlan' in has_trunk and 'native_vlan' not in wants_trunk:
        commands.append('no switchport trunk native vlan')
    return commands