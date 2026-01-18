from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import Connection
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.data_post_processor import (
def filter_system_npu_data(json):
    option_list = ['capwap_offload', 'dedicated_management_affinity', 'dedicated_management_cpu', 'fastpath', 'gtp_enhanced_cpu_range', 'gtp_enhanced_mode', 'intf_shaping_offload', 'ipsec_dec_subengine_mask', 'ipsec_enc_subengine_mask', 'ipsec_inbound_cache', 'ipsec_mtu_override', 'ipsec_over_vlink', 'isf_np_queues', 'lag_out_port_select', 'mcast_session_accounting', 'port_cpu_map', 'port_npu_map', 'priority_protocol', 'qos_mode', 'rdp_offload', 'session_denied_offload', 'sse_backpressure', 'strip_clear_text_padding', 'strip_esp_padding', 'sw_eh_hash', 'sw_np_bandwidth', 'sw_tr_hash', 'uesp_offload']
    json = remove_invalid_fields(json)
    dictionary = {}
    for attribute in option_list:
        if attribute in json and json[attribute] is not None:
            dictionary[attribute] = json[attribute]
    return dictionary