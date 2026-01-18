from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.edgeswitch.edgeswitch import load_config, run_commands
from ansible_collections.community.network.plugins.module_utils.network.edgeswitch.edgeswitch import build_aggregate_spec, map_params_to_obj
from ansible_collections.community.network.plugins.module_utils.network.edgeswitch.edgeswitch_interface import InterfaceConfiguration, merge_interfaces
def parse_vlan_brief(vlan_out):
    have = []
    for line in vlan_out.split('\n'):
        obj = re.match('(?P<vlan_id>\\d+)\\s+(?P<name>[^\\s]+)\\s+', line)
        if obj:
            have.append(obj.groupdict())
    return have