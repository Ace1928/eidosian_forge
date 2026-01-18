from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import Connection
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.data_post_processor import (
def filter_router_ospf_data(json):
    option_list = ['abr_type', 'area', 'auto_cost_ref_bandwidth', 'bfd', 'database_overflow', 'database_overflow_max_lsas', 'database_overflow_time_to_recover', 'default_information_metric', 'default_information_metric_type', 'default_information_originate', 'default_information_route_map', 'default_metric', 'distance', 'distance_external', 'distance_inter_area', 'distance_intra_area', 'distribute_list', 'distribute_list_in', 'distribute_route_map_in', 'log_neighbour_changes', 'neighbor', 'network', 'ospf_interface', 'passive_interface', 'redistribute', 'restart_mode', 'restart_on_topology_change', 'restart_period', 'rfc1583_compatible', 'router_id', 'spf_timers', 'summary_address']
    json = remove_invalid_fields(json)
    dictionary = {}
    for attribute in option_list:
        if attribute in json and json[attribute] is not None:
            dictionary[attribute] = json[attribute]
    return dictionary