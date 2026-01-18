from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import Connection
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.data_post_processor import (
def filter_firewall_access_proxy6_data(json):
    option_list = ['add_vhost_domain_to_dnsdb', 'api_gateway', 'api_gateway6', 'auth_portal', 'auth_virtual_host', 'client_cert', 'decrypted_traffic_mirror', 'empty_cert_action', 'http_supported_max_version', 'log_blocked_traffic', 'name', 'svr_pool_multiplex', 'svr_pool_server_max_concurrent_request', 'svr_pool_server_max_request', 'svr_pool_ttl', 'user_agent_detect', 'vip']
    json = remove_invalid_fields(json)
    dictionary = {}
    for attribute in option_list:
        if attribute in json and json[attribute] is not None:
            dictionary[attribute] = json[attribute]
    return dictionary