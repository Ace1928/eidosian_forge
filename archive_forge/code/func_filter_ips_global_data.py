from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import Connection
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.data_post_processor import (
def filter_ips_global_data(json):
    option_list = ['anomaly_mode', 'cp_accel_mode', 'database', 'deep_app_insp_db_limit', 'deep_app_insp_timeout', 'engine_count', 'exclude_signatures', 'fail_open', 'intelligent_mode', 'ips_reserve_cpu', 'ngfw_max_scan_range', 'np_accel_mode', 'packet_log_queue_depth', 'session_limit_mode', 'skype_client_public_ipaddr', 'socket_size', 'sync_session_ttl', 'tls_active_probe', 'traffic_submit']
    json = remove_invalid_fields(json)
    dictionary = {}
    for attribute in option_list:
        if attribute in json and json[attribute] is not None:
            dictionary[attribute] = json[attribute]
    return dictionary