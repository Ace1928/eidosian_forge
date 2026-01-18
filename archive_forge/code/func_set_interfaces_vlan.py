from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.edgeswitch.edgeswitch import load_config, run_commands
from ansible_collections.community.network.plugins.module_utils.network.edgeswitch.edgeswitch import build_aggregate_spec, map_params_to_obj
from ansible_collections.community.network.plugins.module_utils.network.edgeswitch.edgeswitch_interface import InterfaceConfiguration, merge_interfaces
def set_interfaces_vlan(interfaces_param, interfaces, vlan_id, type):
    """ set vlan_id type for each interface in interfaces_param on interfaces
        unrange interfaces_param if needed
    """
    if interfaces_param:
        for i in interfaces_param:
            match = re.search('(\\d+)\\/(\\d+)-(\\d+)\\/(\\d+)', i)
            if match:
                group = match.group(1)
                start = int(match.group(2))
                end = int(match.group(4))
                for x in range(start, end + 1):
                    key = '{0}/{1}'.format(group, x)
                    interfaces[key].set_vlan(vlan_id, type)
            else:
                interfaces[i].set_vlan(vlan_id, type)