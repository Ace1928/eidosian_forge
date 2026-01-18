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
def filter_firewall_vip6_data(json):
    option_list = ['add_nat64_route', 'arp_reply', 'color', 'comment', 'embedded_ipv4_address', 'extip', 'extport', 'http_cookie_age', 'http_cookie_domain', 'http_cookie_domain_from_host', 'http_cookie_generation', 'http_cookie_path', 'http_cookie_share', 'http_ip_header', 'http_ip_header_name', 'http_multiplex', 'http_redirect', 'https_cookie_secure', 'id', 'ipv4_mappedip', 'ipv4_mappedport', 'ldb_method', 'mappedip', 'mappedport', 'max_embryonic_connections', 'monitor', 'name', 'nat_source_vip', 'nat64', 'nat66', 'ndp_reply', 'outlook_web_access', 'persistence', 'portforward', 'protocol', 'realservers', 'server_type', 'src_filter', 'ssl_accept_ffdhe_groups', 'ssl_algorithm', 'ssl_certificate', 'ssl_cipher_suites', 'ssl_client_fallback', 'ssl_client_rekey_count', 'ssl_client_renegotiation', 'ssl_client_session_state_max', 'ssl_client_session_state_timeout', 'ssl_client_session_state_type', 'ssl_dh_bits', 'ssl_hpkp', 'ssl_hpkp_age', 'ssl_hpkp_backup', 'ssl_hpkp_include_subdomains', 'ssl_hpkp_primary', 'ssl_hpkp_report_uri', 'ssl_hsts', 'ssl_hsts_age', 'ssl_hsts_include_subdomains', 'ssl_http_location_conversion', 'ssl_http_match_host', 'ssl_max_version', 'ssl_min_version', 'ssl_mode', 'ssl_pfs', 'ssl_send_empty_frags', 'ssl_server_algorithm', 'ssl_server_cipher_suites', 'ssl_server_max_version', 'ssl_server_min_version', 'ssl_server_renegotiation', 'ssl_server_session_state_max', 'ssl_server_session_state_timeout', 'ssl_server_session_state_type', 'type', 'uuid', 'weblogic_server', 'websphere_server']
    json = remove_invalid_fields(json)
    dictionary = {}
    for attribute in option_list:
        if attribute in json and json[attribute] is not None:
            dictionary[attribute] = json[attribute]
    return dictionary