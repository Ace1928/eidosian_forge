from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule, json
from ansible_collections.cisco.meraki.plugins.module_utils.network.meraki.meraki import (
def convert_vlan_id(vlan_id):
    if vlan_id == '':
        return None
    elif vlan_id == 0:
        return None
    elif vlan_id in range(1, 4094):
        return vlan_id