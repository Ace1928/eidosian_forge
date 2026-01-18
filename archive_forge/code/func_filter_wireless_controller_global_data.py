from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import Connection
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.data_post_processor import (
def filter_wireless_controller_global_data(json):
    option_list = ['acd_process_count', 'ap_log_server', 'ap_log_server_ip', 'ap_log_server_port', 'control_message_offload', 'data_ethernet_II', 'dfs_lab_test', 'discovery_mc_addr', 'fiapp_eth_type', 'image_download', 'ipsec_base_ip', 'link_aggregation', 'location', 'max_clients', 'max_retransmit', 'mesh_eth_type', 'nac_interval', 'name', 'rogue_scan_mac_adjacency', 'tunnel_mode', 'wpad_process_count', 'wtp_share']
    json = remove_invalid_fields(json)
    dictionary = {}
    for attribute in option_list:
        if attribute in json and json[attribute] is not None:
            dictionary[attribute] = json[attribute]
    return dictionary