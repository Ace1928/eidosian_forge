from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import Connection
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.data_post_processor import (
def filter_router_bgp_data(json):
    option_list = ['additional_path', 'additional_path_select', 'additional_path_select_vpnv4', 'additional_path_select6', 'additional_path_vpnv4', 'additional_path6', 'admin_distance', 'aggregate_address', 'aggregate_address6', 'always_compare_med', 'as', 'bestpath_as_path_ignore', 'bestpath_cmp_confed_aspath', 'bestpath_cmp_routerid', 'bestpath_med_confed', 'bestpath_med_missing_as_worst', 'client_to_client_reflection', 'cluster_id', 'confederation_identifier', 'confederation_peers', 'cross_family_conditional_adv', 'dampening', 'dampening_max_suppress_time', 'dampening_reachability_half_life', 'dampening_reuse', 'dampening_route_map', 'dampening_suppress', 'dampening_unreachability_half_life', 'default_local_preference', 'deterministic_med', 'distance_external', 'distance_internal', 'distance_local', 'ebgp_multipath', 'enforce_first_as', 'fast_external_failover', 'graceful_end_on_timer', 'graceful_restart', 'graceful_restart_time', 'graceful_stalepath_time', 'graceful_update_delay', 'holdtime_timer', 'ibgp_multipath', 'ignore_optional_capability', 'keepalive_timer', 'log_neighbour_changes', 'multipath_recursive_distance', 'neighbor', 'neighbor_group', 'neighbor_range', 'neighbor_range6', 'network', 'network_import_check', 'network6', 'recursive_inherit_priority', 'recursive_next_hop', 'redistribute', 'redistribute6', 'router_id', 'scan_time', 'synchronization', 'tag_resolve_mode', 'vrf', 'vrf_leak', 'vrf_leak6', 'vrf6']
    json = remove_invalid_fields(json)
    dictionary = {}
    for attribute in option_list:
        if attribute in json and json[attribute] is not None:
            dictionary[attribute] = json[attribute]
    return dictionary