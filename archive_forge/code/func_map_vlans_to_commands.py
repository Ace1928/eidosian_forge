from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.edgeswitch.edgeswitch import load_config, run_commands
from ansible_collections.community.network.plugins.module_utils.network.edgeswitch.edgeswitch import build_aggregate_spec, map_params_to_obj
from ansible_collections.community.network.plugins.module_utils.network.edgeswitch.edgeswitch_interface import InterfaceConfiguration, merge_interfaces
def map_vlans_to_commands(want, have, module):
    commands = []
    vlans_added = []
    vlans_removed = []
    vlans_names = []
    for w in want:
        vlan_id = w['vlan_id']
        name = w['name']
        state = w['state']
        obj_in_have = search_obj_in_list(vlan_id, have)
        if state == 'absent':
            if obj_in_have:
                vlans_removed.append(vlan_id)
        elif state == 'present':
            if not obj_in_have:
                vlans_added.append(vlan_id)
                if name:
                    vlans_names.append('vlan name {0} "{1}"'.format(vlan_id, name))
            elif name:
                if name != obj_in_have['name']:
                    vlans_names.append('vlan name {0} "{1}"'.format(vlan_id, name))
    if module.params['purge']:
        for h in have:
            obj_in_want = search_obj_in_list(h['vlan_id'], want)
            if not obj_in_want and h['vlan_id'] != '1':
                vlans_removed.append(h['vlan_id'])
    if vlans_removed:
        commands.append('no vlan {0}'.format(','.join(vlans_removed)))
    if vlans_added:
        commands.append('vlan {0}'.format(','.join(vlans_added)))
    if vlans_names:
        commands.extend(vlans_names)
    if commands:
        commands.insert(0, 'vlan database')
        commands.append('exit')
    return commands