from __future__ import absolute_import, division, print_function
import re
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.argspec.route_maps.route_maps import Route_mapsArgs
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import remove_empties_from_list
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic \
def get_all_route_maps(self):
    """Execute a REST "GET" API to fetch all of the current route map configuration
        from the target device."""
    route_map_fetch_spec = 'openconfig-routing-policy:routing-policy/policy-definitions'
    route_map_resp_key = 'openconfig-routing-policy:policy-definitions'
    route_map_key = 'policy-definition'
    url = 'data/%s' % route_map_fetch_spec
    method = 'GET'
    request = [{'path': url, 'method': method}]
    try:
        response = edit_config(self._module, to_request(self._module, request))
    except ConnectionError as exc:
        self._module.fail_json(msg=str(exc))
    route_maps_unparsed = []
    resp_route_map_envelope = response[0][1].get(route_map_resp_key, None)
    if resp_route_map_envelope:
        route_maps_unparsed = resp_route_map_envelope.get(route_map_key, None)
    return route_maps_unparsed