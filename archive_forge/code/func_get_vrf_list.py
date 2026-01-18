from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
def get_vrf_list(module):
    try:
        body = run_commands(module, ['show vrf all | json'])[0]
        vrf_table = body['TABLE_vrf']['ROW_vrf']
    except (KeyError, AttributeError):
        return []
    vrf_list = []
    if vrf_table:
        for each in vrf_table:
            vrf_list.append(str(each['vrf_name'].lower()))
    return vrf_list