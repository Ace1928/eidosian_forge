from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import Connection
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.data_post_processor import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.comparison import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.comparison import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.comparison import (
def filter_vpn_ipsec_phase2_data(json):
    option_list = ['add_route', 'auto_negotiate', 'comments', 'dhcp_ipsec', 'dhgrp', 'diffserv', 'diffservcode', 'dst_addr_type', 'dst_end_ip', 'dst_end_ip6', 'dst_name', 'dst_name6', 'dst_port', 'dst_start_ip', 'dst_start_ip6', 'dst_subnet', 'dst_subnet6', 'encapsulation', 'inbound_dscp_copy', 'initiator_ts_narrow', 'ipv4_df', 'keepalive', 'keylife_type', 'keylifekbs', 'keylifeseconds', 'l2tp', 'name', 'pfs', 'phase1name', 'proposal', 'protocol', 'replay', 'route_overlap', 'selector_match', 'single_source', 'src_addr_type', 'src_end_ip', 'src_end_ip6', 'src_name', 'src_name6', 'src_port', 'src_start_ip', 'src_start_ip6', 'src_subnet', 'src_subnet6', 'use_natip']
    json = remove_invalid_fields(json)
    dictionary = {}
    for attribute in option_list:
        if attribute in json and json[attribute] is not None:
            dictionary[attribute] = json[attribute]
    return dictionary