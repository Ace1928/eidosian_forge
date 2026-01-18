from __future__ import (absolute_import, division, print_function)
import json
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import \
def get_vlan_ids(rest_obj):
    resp = rest_obj.invoke_request('GET', VLANS)
    vlans = resp.json_data.get('value')
    vlan_map = {}
    natives = {}
    for vlan in vlans:
        vlan_map[vlan['Name']] = vlan['Id']
        if vlan['VlanMaximum'] == vlan['VlanMinimum']:
            natives[vlan['VlanMaximum']] = vlan['Id']
    natives.update({0: 0})
    return (vlan_map, natives)