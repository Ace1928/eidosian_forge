from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule, json
from ansible_collections.cisco.meraki.plugins.module_utils.network.meraki.meraki import (
def does_route_exist(name, routes):
    for route in routes:
        if name == route['name']:
            return route
    return None