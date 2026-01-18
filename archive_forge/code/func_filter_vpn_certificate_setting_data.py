from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import Connection
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.data_post_processor import (
def filter_vpn_certificate_setting_data(json):
    option_list = ['cert_expire_warning', 'certname_dsa1024', 'certname_dsa2048', 'certname_ecdsa256', 'certname_ecdsa384', 'certname_ecdsa521', 'certname_ed25519', 'certname_ed448', 'certname_rsa1024', 'certname_rsa2048', 'certname_rsa4096', 'check_ca_cert', 'check_ca_chain', 'cmp_key_usage_checking', 'cmp_save_extra_certs', 'cn_allow_multi', 'cn_match', 'crl_verification', 'interface', 'interface_select_method', 'ocsp_default_server', 'ocsp_option', 'ocsp_status', 'proxy', 'proxy_password', 'proxy_port', 'proxy_username', 'source_ip', 'ssl_min_proto_version', 'ssl_ocsp_option', 'ssl_ocsp_source_ip', 'ssl_ocsp_status', 'strict_crl_check', 'strict_ocsp_check', 'subject_match', 'subject_set']
    json = remove_invalid_fields(json)
    dictionary = {}
    for attribute in option_list:
        if attribute in json and json[attribute] is not None:
            dictionary[attribute] = json[attribute]
    return dictionary