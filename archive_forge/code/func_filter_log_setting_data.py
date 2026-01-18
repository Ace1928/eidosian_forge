from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import Connection
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.data_post_processor import (
def filter_log_setting_data(json):
    option_list = ['anonymization_hash', 'brief_traffic_format', 'custom_log_fields', 'daemon_log', 'expolicy_implicit_log', 'extended_log', 'faz_override', 'fortiview_weekly_data', 'fwpolicy_implicit_log', 'fwpolicy6_implicit_log', 'local_in_allow', 'local_in_deny_broadcast', 'local_in_deny_unicast', 'local_out', 'local_out_ioc_detection', 'log_invalid_packet', 'log_policy_comment', 'log_policy_name', 'log_user_in_upper', 'neighbor_event', 'resolve_ip', 'resolve_port', 'rest_api_get', 'rest_api_set', 'syslog_override', 'user_anonymize']
    json = remove_invalid_fields(json)
    dictionary = {}
    for attribute in option_list:
        if attribute in json and json[attribute] is not None:
            dictionary[attribute] = json[attribute]
    return dictionary