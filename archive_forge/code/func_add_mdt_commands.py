from __future__ import absolute_import, division, print_function
import re
import time
from functools import partial
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import exec_command
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.config import (
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.ios import (
def add_mdt_commands(afi_dict, vrf_name, commands):
    for key, value in afi_dict['mdt'].items():
        cmd = KEY_TO_COMMAND_MAP[key]
        if key in ['default', 'data_mcast']:
            cmd = cmd + value['vxlan_mcast_group']
            add_command_to_vrf(vrf_name, cmd, commands)
        elif key == 'data_threshold':
            cmd = cmd + str(value)
            add_command_to_vrf(vrf_name, cmd, commands)
        elif key == 'auto_discovery':
            if value['vxlan']['enable']:
                cmd = cmd + 'vxlan'
            if value['vxlan'].get('inter_as'):
                cmd = cmd + ' ' + 'inter-as'
            add_command_to_vrf(vrf_name, cmd, commands)
        elif key == 'overlay':
            if value['use_bgp']['enable']:
                cmd = cmd + 'use-bgp'
            if value['use_bgp'].get('spt_only'):
                cmd = cmd + ' ' + 'spt-only'
            add_command_to_vrf(vrf_name, cmd, commands)