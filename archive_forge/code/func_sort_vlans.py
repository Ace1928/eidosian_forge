from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule, json
from ansible_collections.cisco.meraki.plugins.module_utils.network.meraki.meraki import (
def sort_vlans(meraki, vlans):
    converted = set()
    for vlan in vlans:
        converted.add(int(vlan))
    vlans_sorted = sorted(converted)
    vlans_str = []
    for vlan in vlans_sorted:
        vlans_str.append(str(vlan))
    return ','.join(vlans_str)