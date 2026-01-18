from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import Connection
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.data_post_processor import (
def filter_web_proxy_global_data(json):
    option_list = ['fast_policy_match', 'forward_proxy_auth', 'forward_server_affinity_timeout', 'ldap_user_cache', 'learn_client_ip', 'learn_client_ip_from_header', 'learn_client_ip_srcaddr', 'learn_client_ip_srcaddr6', 'log_forward_server', 'max_message_length', 'max_request_length', 'max_waf_body_cache_length', 'proxy_fqdn', 'src_affinity_exempt_addr', 'src_affinity_exempt_addr6', 'ssl_ca_cert', 'ssl_cert', 'strict_web_check', 'tunnel_non_http', 'unknown_http_version', 'webproxy_profile']
    json = remove_invalid_fields(json)
    dictionary = {}
    for attribute in option_list:
        if attribute in json and json[attribute] is not None:
            dictionary[attribute] = json[attribute]
    return dictionary