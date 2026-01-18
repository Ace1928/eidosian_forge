from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import Connection
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.data_post_processor import (
def filter_firewall_gtp_data(json):
    option_list = ['addr_notify', 'apn', 'apn_filter', 'authorized_ggsns', 'authorized_ggsns6', 'authorized_sgsns', 'authorized_sgsns6', 'comment', 'context_id', 'control_plane_message_rate_limit', 'default_apn_action', 'default_imsi_action', 'default_ip_action', 'default_noip_action', 'default_policy_action', 'denied_log', 'echo_request_interval', 'extension_log', 'forwarded_log', 'global_tunnel_limit', 'gtp_in_gtp', 'gtpu_denied_log', 'gtpu_forwarded_log', 'gtpu_log_freq', 'half_close_timeout', 'half_open_timeout', 'handover_group', 'handover_group6', 'ie_allow_list_v0v1', 'ie_allow_list_v2', 'ie_remove_policy', 'ie_remover', 'ie_validation', 'ie_white_list_v0v1', 'ie_white_list_v2', 'imsi', 'imsi_filter', 'interface_notify', 'invalid_reserved_field', 'invalid_sgsns_to_log', 'invalid_sgsns6_to_log', 'ip_filter', 'ip_policy', 'log_freq', 'log_gtpu_limit', 'log_imsi_prefix', 'log_msisdn_prefix', 'max_message_length', 'message_filter_v0v1', 'message_filter_v2', 'message_rate_limit', 'message_rate_limit_v0', 'message_rate_limit_v1', 'message_rate_limit_v2', 'min_message_length', 'miss_must_ie', 'monitor_mode', 'name', 'noip_filter', 'noip_policy', 'out_of_state_ie', 'out_of_state_message', 'per_apn_shaper', 'policy', 'policy_filter', 'policy_v2', 'port_notify', 'rat_timeout_profile', 'rate_limit_mode', 'rate_limited_log', 'rate_sampling_interval', 'remove_if_echo_expires', 'remove_if_recovery_differ', 'reserved_ie', 'send_delete_when_timeout', 'send_delete_when_timeout_v2', 'spoof_src_addr', 'state_invalid_log', 'sub_second_interval', 'sub_second_sampling', 'traffic_count_log', 'tunnel_limit', 'tunnel_limit_log', 'tunnel_timeout', 'unknown_version_action', 'user_plane_message_rate_limit', 'warning_threshold']
    json = remove_invalid_fields(json)
    dictionary = {}
    for attribute in option_list:
        if attribute in json and json[attribute] is not None:
            dictionary[attribute] = json[attribute]
    return dictionary