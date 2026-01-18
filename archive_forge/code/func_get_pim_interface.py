from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
def get_pim_interface(module, interface):
    pim_interface = {}
    body = get_config(module, flags=['interface {0}'.format(interface)])
    pim_interface['bfd'] = 'default'
    pim_interface['neighbor_type'] = None
    pim_interface['neighbor_policy'] = None
    pim_interface['jp_policy_in'] = None
    pim_interface['jp_policy_out'] = None
    pim_interface['jp_type_in'] = None
    pim_interface['jp_type_out'] = None
    pim_interface['jp_bidir'] = False
    pim_interface['isauth'] = False
    if body:
        all_lines = body.splitlines()
        for each in all_lines:
            if 'jp-policy' in each:
                policy_name = re.search('ip pim jp-policy(?: prefix-list)? (\\S+)(?: \\S+)?', each).group(1)
                if 'prefix-list' in each:
                    ptype = 'prefix'
                else:
                    ptype = 'routemap'
                if 'out' in each:
                    pim_interface['jp_policy_out'] = policy_name
                    pim_interface['jp_type_out'] = ptype
                elif 'in' in each:
                    pim_interface['jp_policy_in'] = policy_name
                    pim_interface['jp_type_in'] = ptype
                else:
                    pim_interface['jp_policy_in'] = policy_name
                    pim_interface['jp_policy_out'] = policy_name
                    pim_interface['jp_bidir'] = True
            elif 'neighbor-policy' in each:
                pim_interface['neighbor_policy'] = re.search('ip pim neighbor-policy(?: prefix-list)? (\\S+)', each).group(1)
                if 'prefix-list' in each:
                    pim_interface['neighbor_type'] = 'prefix'
                else:
                    pim_interface['neighbor_type'] = 'routemap'
            elif 'ah-md5' in each:
                pim_interface['isauth'] = True
            elif 'sparse-mode' in each:
                pim_interface['sparse'] = True
            elif 'bfd-instance' in each:
                m = re.search('ip pim bfd-instance(?P<disable> disable)?', each)
                if m:
                    pim_interface['bfd'] = 'disable' if m.group('disable') else 'enable'
            elif 'border' in each:
                pim_interface['border'] = True
            elif 'hello-interval' in each:
                pim_interface['hello_interval'] = re.search('ip pim hello-interval (\\d+)', body).group(1)
            elif 'dr-priority' in each:
                pim_interface['dr_prio'] = re.search('ip pim dr-priority (\\d+)', body).group(1)
    return pim_interface