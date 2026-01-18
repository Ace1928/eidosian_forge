from __future__ import absolute_import, division, print_function
import re
from copy import deepcopy
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import remove_default_spec
from ansible_collections.community.network.plugins.module_utils.network.slxos.slxos import get_config, load_config, run_commands
def get_list_of_vlans(module):
    config = run_commands(module, ['show vlan brief'])[0]
    vlans = set()
    lines = config.strip().splitlines()
    for line in lines:
        line_parts = line.split()
        if line_parts:
            try:
                int(line_parts[0])
            except ValueError:
                continue
            vlans.add(line_parts[0])
    return list(vlans)