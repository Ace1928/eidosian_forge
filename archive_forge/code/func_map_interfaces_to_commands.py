from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.edgeswitch.edgeswitch import load_config, run_commands
from ansible_collections.community.network.plugins.module_utils.network.edgeswitch.edgeswitch import build_aggregate_spec, map_params_to_obj
from ansible_collections.community.network.plugins.module_utils.network.edgeswitch.edgeswitch_interface import InterfaceConfiguration, merge_interfaces
def map_interfaces_to_commands(want, ports, module):
    commands = list()
    interfaces = {}
    for key, value in ports.items():
        interfaces[key] = VlanInterfaceConfiguration()
    for w in want:
        state = w['state']
        if state != 'present':
            continue
        auto_tag = w['auto_tag']
        auto_untag = w['auto_untag']
        auto_exclude = w['auto_exclude']
        vlan_id = w['vlan_id']
        tagged_interfaces = w['tagged_interfaces']
        untagged_interfaces = w['untagged_interfaces']
        excluded_interfaces = w['excluded_interfaces']
        for key, value in ports.items():
            if auto_tag:
                interfaces[key].tagged.append(vlan_id)
            elif auto_exclude:
                interfaces[key].excluded.append(vlan_id)
            elif auto_untag:
                interfaces[key].untagged.append(vlan_id)
        set_interfaces_vlan(tagged_interfaces, interfaces, vlan_id, 'tagged')
        set_interfaces_vlan(untagged_interfaces, interfaces, vlan_id, 'untagged')
        set_interfaces_vlan(excluded_interfaces, interfaces, vlan_id, 'excluded')
    for i, interface in interfaces.items():
        port = ports[i]
        interface.gen_commands(port, module)
    interfaces = merge_interfaces(interfaces)
    for i, interface in interfaces.items():
        if len(interface.commands) > 0:
            commands.append('interface {0}'.format(i))
            commands.extend(interface.commands)
    return commands