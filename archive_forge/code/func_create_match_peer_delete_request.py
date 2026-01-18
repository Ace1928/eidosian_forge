from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts \
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils \
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
def create_match_peer_delete_request(self, route_map_payload, peer_str):
    """Create a request to delete the current "match peer" configuration for the
        route map statement corresponding to the incoming route map update request
        specified by the "route_map_payload," input parameter. Return the created request."""
    if not route_map_payload:
        return {}
    conf_map_name = route_map_payload.get('name')
    conf_seq_num = route_map_payload['statements']['statement'][0]['name']
    if not conf_map_name or not conf_seq_num:
        return {}
    match_delete_req_base = self.route_map_stmt_base_uri.format(conf_map_name, conf_seq_num) + 'conditions/'
    request_uri = match_delete_req_base + 'match-neighbor-set/config/openconfig-routing-policy-ext:address={0}'.format(peer_str)
    request = {'path': request_uri, 'method': DELETE}
    return request