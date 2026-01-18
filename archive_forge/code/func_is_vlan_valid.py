from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.meraki.plugins.module_utils.network.meraki.meraki import MerakiModule, meraki_argument_spec
import json
def is_vlan_valid(meraki, net_id, vlan_id):
    vlans = get_vlans(meraki, net_id)
    for vlan in vlans:
        if vlan_id == vlan['id']:
            return True
    return False