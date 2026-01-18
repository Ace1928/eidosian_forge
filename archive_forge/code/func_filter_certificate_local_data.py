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
def filter_certificate_local_data(json):
    option_list = ['acme_ca_url', 'acme_domain', 'acme_email', 'acme_renew_window', 'acme_rsa_key_size', 'auto_regenerate_days', 'auto_regenerate_days_warning', 'ca_identifier', 'certificate', 'cmp_path', 'cmp_regeneration_method', 'cmp_server', 'cmp_server_cert', 'comments', 'csr', 'enroll_protocol', 'est_ca_id', 'est_client_cert', 'est_http_password', 'est_http_username', 'est_server', 'est_server_cert', 'est_srp_password', 'est_srp_username', 'ike_localid', 'ike_localid_type', 'last_updated', 'name', 'name_encoding', 'password', 'private_key', 'private_key_retain', 'range', 'scep_password', 'scep_url', 'source', 'source_ip', 'state']
    json = remove_invalid_fields(json)
    dictionary = {}
    for attribute in option_list:
        if attribute in json and json[attribute] is not None:
            dictionary[attribute] = json[attribute]
    return dictionary