from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import Connection
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.data_post_processor import (
def filter_log_fortianalyzer3_setting_data(json):
    option_list = ['__change_ip', 'access_config', 'alt_server', 'certificate', 'certificate_verification', 'conn_timeout', 'enc_algorithm', 'fallback_to_primary', 'faz_type', 'hmac_algorithm', 'interface', 'interface_select_method', 'ips_archive', 'max_log_rate', 'mgmt_name', 'monitor_failure_retry_period', 'monitor_keepalive_period', 'preshared_key', 'priority', 'reliable', 'serial', 'server', 'source_ip', 'ssl_min_proto_version', 'status', 'upload_day', 'upload_interval', 'upload_option', 'upload_time']
    json = remove_invalid_fields(json)
    dictionary = {}
    for attribute in option_list:
        if attribute in json and json[attribute] is not None:
            dictionary[attribute] = json[attribute]
    return dictionary