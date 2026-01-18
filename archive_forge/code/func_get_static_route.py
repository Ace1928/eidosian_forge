from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule, json
from ansible_collections.cisco.meraki.plugins.module_utils.network.meraki.meraki import (
def get_static_route(meraki, net_id, route_id):
    path = meraki.construct_path('get_one', net_id=net_id, custom={'route_id': route_id})
    r = meraki.request(path, method='GET')
    return r