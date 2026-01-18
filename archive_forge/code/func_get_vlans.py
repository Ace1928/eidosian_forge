from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.meraki.plugins.module_utils.network.meraki.meraki import MerakiModule, meraki_argument_spec
import json
def get_vlans(meraki, net_id):
    path = meraki.construct_path('get_all', net_id=net_id)
    return meraki.request(path, method='GET')