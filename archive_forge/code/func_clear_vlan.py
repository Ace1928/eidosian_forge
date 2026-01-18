from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule, json
from ansible_collections.cisco.meraki.plugins.module_utils.network.meraki.meraki import (
def clear_vlan(params, payload):
    if params['vlan'] == 0:
        payload['vlan'] = None
    return payload