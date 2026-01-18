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
def filter_gtp_message_filter_v0v1_data(json):
    option_list = ['create_mbms', 'create_pdp', 'data_record', 'delete_aa_pdp', 'delete_mbms', 'delete_pdp', 'echo', 'end_marker', 'error_indication', 'failure_report', 'fwd_relocation', 'fwd_srns_context', 'gtp_pdu', 'identification', 'mbms_de_registration', 'mbms_notification', 'mbms_registration', 'mbms_session_start', 'mbms_session_stop', 'mbms_session_update', 'ms_info_change_notif', 'name', 'node_alive', 'note_ms_present', 'pdu_notification', 'ran_info', 'redirection', 'relocation_cancel', 'send_route', 'sgsn_context', 'support_extension', 'ue_registration_query', 'unknown_message', 'unknown_message_white_list', 'update_mbms', 'update_pdp', 'v0_create_aa_pdp__v1_init_pdp_ctx', 'version_not_support']
    json = remove_invalid_fields(json)
    dictionary = {}
    for attribute in option_list:
        if attribute in json and json[attribute] is not None:
            dictionary[attribute] = json[attribute]
    return dictionary