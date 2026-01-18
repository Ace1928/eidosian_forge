from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts \
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils \
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
def route_map_remove_configured_match_peer(self, route_map_payload, have, requests):
    """If a route map "match peer" condition is configured in the route map
        statement corresponding to the incoming route map update request
        specified by the "route_map_payload" input parameter, equeue a REST API request
        to delete it."""
    if route_map_payload['statements']['statement'][0].get('conditions') and route_map_payload['statements']['statement'][0]['conditions'].get('match-neighbor-set'):
        peer = self.match_peer_configured(route_map_payload, have)
        if peer:
            request = self.create_match_peer_delete_request(route_map_payload, peer)
            if request:
                requests.append(request)