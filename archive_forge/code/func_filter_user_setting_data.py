from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import Connection
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.data_post_processor import (
def filter_user_setting_data(json):
    option_list = ['auth_blackout_time', 'auth_ca_cert', 'auth_cert', 'auth_http_basic', 'auth_invalid_max', 'auth_lockout_duration', 'auth_lockout_threshold', 'auth_on_demand', 'auth_portal_timeout', 'auth_ports', 'auth_secure_http', 'auth_src_mac', 'auth_ssl_allow_renegotiation', 'auth_ssl_max_proto_version', 'auth_ssl_min_proto_version', 'auth_ssl_sigalgs', 'auth_timeout', 'auth_timeout_type', 'auth_type', 'default_user_password_policy', 'per_policy_disclaimer', 'radius_ses_timeout_act']
    json = remove_invalid_fields(json)
    dictionary = {}
    for attribute in option_list:
        if attribute in json and json[attribute] is not None:
            dictionary[attribute] = json[attribute]
    return dictionary