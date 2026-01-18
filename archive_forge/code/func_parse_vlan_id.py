from __future__ import absolute_import, division, print_function
import re
from time import sleep
import itertools
from copy import deepcopy
from time import sleep
from ansible.module_utils._text import to_text
from ansible.module_utils.basic import AnsibleModule, env_fallback
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.config import NetworkConfig
from ansible_collections.community.network.plugins.module_utils.network.icx.icx import load_config, get_config
from ansible.module_utils.connection import Connection, ConnectionError, exec_command
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import conditional, remove_default_spec
def parse_vlan_id(module):
    vlans = []
    command = 'show vlan brief'
    rc, out, err = exec_command(module, command)
    lines = out.split('\n')
    for line in lines:
        if 'VLANs Configured :' in line:
            values = line.split(':')[1]
            vlans = [s for s in values.split() if s.isdigit()]
            s = re.findall('(?P<low>\\d+)\\sto\\s(?P<high>\\d+)', values)
            for ranges in s:
                low = int(ranges[0]) + 1
                high = int(ranges[1])
                while high > low:
                    vlans.append(str(low))
                    low = low + 1
    return vlans