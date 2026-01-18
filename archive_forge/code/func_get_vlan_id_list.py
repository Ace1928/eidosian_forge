from __future__ import absolute_import, division, print_function
import re
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible.module_utils.connection import ConnectionError
@staticmethod
def get_vlan_id_list(vlan_range_list):
    """Returns a list of all VLAN IDs specified in VLAN range list"""
    vlan_id_list = []
    if vlan_range_list:
        for vlan_range in vlan_range_list:
            vlan_val = vlan_range['vlan']
            if '-' in vlan_val:
                match = re.match('Vlan(\\d+)-(\\d+)', vlan_val)
                if match:
                    vlan_id_list.extend(range(int(match.group(1)), int(match.group(2)) + 1))
            else:
                match = re.match('Vlan(\\d+)', vlan_val)
                if match:
                    vlan_id_list.append(int(match.group(1)))
    return vlan_id_list