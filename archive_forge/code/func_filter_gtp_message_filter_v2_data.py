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
def filter_gtp_message_filter_v2_data(json):
    option_list = ['alert_mme_notif_ack', 'bearer_resource_cmd_fail', 'change_notification', 'configuration_transfer_tunnel', 'context_req_res_ack', 'create_bearer', 'create_forwarding_tunnel_req_resp', 'create_indirect_forwarding_tunnel_req_resp', 'create_session', 'cs_paging', 'delete_bearer_cmd_fail', 'delete_bearer_req_resp', 'delete_indirect_forwarding_tunnel_req_resp', 'delete_pdn_connection_set', 'delete_session', 'detach_notif_ack', 'dlink_data_notif_ack', 'dlink_notif_failure', 'echo', 'forward_access_notif_ack', 'forward_relocation_cmp_notif_ack', 'forward_relocation_req_res', 'identification_req_resp', 'isr_status', 'mbms_session_start_req_resp', 'mbms_session_stop_req_resp', 'mbms_session_update_req_resp', 'modify_access_req_resp', 'modify_bearer_cmd_fail', 'modify_bearer_req_resp', 'name', 'pgw_dlink_notif_ack', 'pgw_restart_notif_ack', 'ran_info_relay', 'release_access_bearer_req_resp', 'relocation_cancel_req_resp', 'remote_ue_report_notif_ack', 'reserved_for_earlier_version', 'resume', 'stop_paging_indication', 'suspend', 'trace_session', 'ue_activity_notif_ack', 'ue_registration_query_req_resp', 'unknown_message', 'unknown_message_white_list', 'update_bearer', 'update_pdn_connection_set', 'version_not_support']
    json = remove_invalid_fields(json)
    dictionary = {}
    for attribute in option_list:
        if attribute in json and json[attribute] is not None:
            dictionary[attribute] = json[attribute]
    return dictionary