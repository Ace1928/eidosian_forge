from __future__ import absolute_import, division, print_function
import re
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.argspec.route_maps.route_maps import Route_mapsArgs
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import remove_empties_from_list
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic \
def get_route_map_stmt_match_attr(self, route_map_stmt, parsed_route_map_stmt):
    """Parse the "match" attributes in the raw input configuration JSON
        representation for the route map "statement" specified
        by the "route_map_stmt," input parameter. Parse the information to
        convert it to a dictionary matching the "argspec" for the "route_maps" resource
        module."""
    parsed_route_map_stmt['match'] = {}
    parsed_rmap_match = parsed_route_map_stmt['match']
    stmt_conditions = route_map_stmt.get('conditions')
    if not stmt_conditions:
        return
    if stmt_conditions.get('match-as-path-set') and stmt_conditions['match-as-path-set'].get('config'):
        as_path = stmt_conditions['match-as-path-set']['config'].get('as-path-set')
        if as_path:
            parsed_rmap_match['as_path'] = as_path
    rmap_bgp_policy_match = stmt_conditions.get('openconfig-bgp-policy:bgp-conditions')
    if rmap_bgp_policy_match:
        self.get_rmap_match_bgp_policy_attr(rmap_bgp_policy_match, parsed_rmap_match)
    if stmt_conditions.get('match-interface') and stmt_conditions['match-interface'].get('config'):
        match_interface = stmt_conditions['match-interface']['config'].get('interface')
        if match_interface:
            parsed_rmap_match['interface'] = match_interface
    if stmt_conditions.get('match-prefix-set') and stmt_conditions['match-prefix-set']['config']:
        match_prefix_set = stmt_conditions['match-prefix-set']['config']
        if match_prefix_set and match_prefix_set.get('prefix-set'):
            if not parsed_rmap_match.get('ip'):
                parsed_rmap_match['ip'] = {}
            parsed_rmap_match['ip']['address'] = match_prefix_set['prefix-set']
        if match_prefix_set and match_prefix_set.get('openconfig-routing-policy-ext:ipv6-prefix-set'):
            parsed_rmap_match['ipv6'] = {}
            parsed_rmap_match['ipv6']['address'] = match_prefix_set['openconfig-routing-policy-ext:ipv6-prefix-set']
        if stmt_conditions.get('match-neighbor-set') and stmt_conditions['match-neighbor-set'].get('config') and stmt_conditions['match-neighbor-set']['config'].get('openconfig-routing-policy-ext:address'):
            parsed_rmap_match_peer = stmt_conditions['match-neighbor-set']['config']['openconfig-routing-policy-ext:address'][0]
            parsed_rmap_match['peer'] = {}
            if ':' in parsed_rmap_match_peer:
                parsed_rmap_match['peer']['ipv6'] = parsed_rmap_match_peer
            elif '.' in parsed_rmap_match_peer:
                parsed_rmap_match['peer']['ip'] = parsed_rmap_match_peer
            else:
                parsed_rmap_match['peer']['interface'] = parsed_rmap_match_peer
    if stmt_conditions.get('config') and stmt_conditions['config'].get('install-protocol-eq'):
        parsed_rmap_match_source_protocol = stmt_conditions['config']['install-protocol-eq']
        if parsed_rmap_match_source_protocol == 'openconfig-policy-types:BGP':
            parsed_rmap_match['source_protocol'] = 'bgp'
        elif parsed_rmap_match_source_protocol == 'openconfig-policy-types:OSPF':
            parsed_rmap_match['source_protocol'] = 'ospf'
        elif parsed_rmap_match_source_protocol == 'openconfig-policy-types:STATIC':
            parsed_rmap_match['source_protocol'] = 'static'
        elif parsed_rmap_match_source_protocol == 'openconfig-policy-types:DIRECTLY_CONNECTED':
            parsed_rmap_match['source_protocol'] = 'connected'
    if stmt_conditions.get('openconfig-routing-policy-ext:match-src-network-instance'):
        match_src_vrf = stmt_conditions['openconfig-routing-policy-ext:match-src-network-instance'].get('config')
        if match_src_vrf and match_src_vrf.get('name'):
            parsed_rmap_match['source_vrf'] = match_src_vrf['name']
    if stmt_conditions.get('match-tag-set') and stmt_conditions['match-tag-set'].get('config'):
        match_tag = stmt_conditions['match-tag-set']['config'].get('openconfig-routing-policy-ext:tag-value')
        if match_tag:
            parsed_rmap_match['tag'] = match_tag[0]