from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.argspec.bgp_neighbors_af.bgp_neighbors_af import Bgp_neighbors_afArgs
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.bgp_utils import (
def fill_route_map(self, data):
    for route_map_key in ['out_route_name', 'in_route_name']:
        if route_map_key in data:
            route_map = data['route_map']
            for e_route in data[route_map_key]:
                direction = route_map_key.split('_', maxsplit=1)[0]
                route_map.append({'name': e_route, 'direction': direction})
            data.pop(route_map_key)