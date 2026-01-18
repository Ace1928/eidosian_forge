from __future__ import absolute_import, division, print_function
import re
import time
from copy import deepcopy
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
def get_existing_vrfs(module):
    objs = list()
    command = 'show vrf all'
    try:
        body = execute_show_command(command, module)[0]
    except IndexError:
        return list()
    try:
        vrf_table = body['TABLE_vrf']['ROW_vrf']
    except (TypeError, IndexError, KeyError):
        return list()
    if isinstance(vrf_table, list):
        for vrf in vrf_table:
            obj = {}
            obj['name'] = vrf['vrf_name']
            objs.append(obj)
    elif isinstance(vrf_table, dict):
        obj = {}
        obj['name'] = vrf_table['vrf_name']
        objs.append(obj)
    return objs