from __future__ import absolute_import, division, print_function
import copy
from ansible.module_utils.basic import AnsibleModule, json
from ansible_collections.cisco.meraki.plugins.module_utils.network.meraki.meraki import MerakiModule, meraki_argument_spec
def get_applications(meraki, net_id):
    path = meraki.construct_path('get_categories', net_id=net_id)
    return meraki.request(path, method='GET')