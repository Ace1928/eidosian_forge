from __future__ import absolute_import, division, print_function
import re
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.argspec.route_maps.route_maps import Route_mapsArgs
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import remove_empties_from_list
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic \
def get_route_map_set_bgp_policy_attr(self, set_bgp_policy, parsed_route_map_stmt_set):
    """Parse the BGP policy "set" attribute portion of the raw input
        configuration JSON representation within the route map "statement"
        that is currently being parsed. The configuration section to be parsed
        is specified by the "set_bgp_policy" input parameter. Parse the
        information to convert it to a dictionary matching the "argspec" for
        the "route_maps" resource module."""
    set_as_path_top = set_bgp_policy.get('set-as-path-prepend')
    if set_as_path_top and set_as_path_top.get('config'):
        as_path_prepend = set_as_path_top['config'].get('openconfig-routing-policy-ext:asn-list')
        if as_path_prepend:
            parsed_route_map_stmt_set['as_path_prepend'] = as_path_prepend
    set_comm_list_delete_top = set_bgp_policy.get('set-community-delete')
    if set_comm_list_delete_top:
        set_comm_list_delete_config = set_comm_list_delete_top.get('config')
        if set_comm_list_delete_config:
            comm_list_delete = set_comm_list_delete_config.get('community-set-delete')
            if comm_list_delete:
                parsed_route_map_stmt_set['comm_list_delete'] = comm_list_delete
    self.get_rmap_set_community(set_bgp_policy, parsed_route_map_stmt_set)
    self.get_rmap_set_extcommunity(set_bgp_policy, parsed_route_map_stmt_set)
    set_bgp_policy_cfg = set_bgp_policy.get('config')
    if set_bgp_policy_cfg:
        ip_next_hop = set_bgp_policy_cfg.get('set-next-hop')
        if ip_next_hop:
            parsed_route_map_stmt_set['ip_next_hop'] = ip_next_hop
        ipv6_next_hop_global_addr = set_bgp_policy_cfg.get('set-ipv6-next-hop-global')
        ipv6_prefer_global = set_bgp_policy_cfg.get('set-ipv6-next-hop-prefer-global')
        if ipv6_next_hop_global_addr or ipv6_prefer_global is not None:
            parsed_route_map_stmt_set['ipv6_next_hop'] = {}
            set_ipv6_nexthop = parsed_route_map_stmt_set['ipv6_next_hop']
            if ipv6_next_hop_global_addr:
                set_ipv6_nexthop['global_addr'] = ipv6_next_hop_global_addr
            if ipv6_prefer_global is not None:
                set_ipv6_nexthop['prefer_global'] = ipv6_prefer_global
        local_preference = set_bgp_policy_cfg.get('set-local-pref')
        if local_preference:
            parsed_route_map_stmt_set['local_preference'] = local_preference
        set_origin = set_bgp_policy_cfg.get('set-route-origin')
        if set_origin:
            if set_origin == 'EGP':
                parsed_route_map_stmt_set['origin'] = 'egp'
            elif set_origin == 'IGP':
                parsed_route_map_stmt_set['origin'] = 'igp'
            elif set_origin == 'INCOMPLETE':
                parsed_route_map_stmt_set['origin'] = 'incomplete'
        weight = set_bgp_policy_cfg.get('set-weight')
        if weight:
            parsed_route_map_stmt_set['weight'] = weight