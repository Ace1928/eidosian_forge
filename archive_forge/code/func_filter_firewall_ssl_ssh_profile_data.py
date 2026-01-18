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
def filter_firewall_ssl_ssh_profile_data(json):
    option_list = ['allowlist', 'block_blacklisted_certificates', 'block_blocklisted_certificates', 'caname', 'comment', 'dot', 'ftps', 'https', 'imaps', 'mapi_over_https', 'name', 'pop3s', 'rpc_over_https', 'server_cert', 'server_cert_mode', 'smtps', 'ssh', 'ssl', 'ssl_anomalies_log', 'ssl_anomaly_log', 'ssl_exempt', 'ssl_exemption_ip_rating', 'ssl_exemption_log', 'ssl_exemptions_log', 'ssl_handshake_log', 'ssl_negotiation_log', 'ssl_server', 'ssl_server_cert_log', 'supported_alpn', 'untrusted_caname', 'use_ssl_server', 'whitelist']
    json = remove_invalid_fields(json)
    dictionary = {}
    for attribute in option_list:
        if attribute in json and json[attribute] is not None:
            dictionary[attribute] = json[attribute]
    return dictionary