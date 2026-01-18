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
def filter_user_ldap_data(json):
    option_list = ['account_key_cert_field', 'account_key_filter', 'account_key_processing', 'account_key_upn_san', 'antiphish', 'ca_cert', 'client_cert', 'client_cert_auth', 'cnid', 'dn', 'group_filter', 'group_member_check', 'group_object_filter', 'group_search_base', 'interface', 'interface_select_method', 'member_attr', 'name', 'obtain_user_info', 'password', 'password_attr', 'password_expiry_warning', 'password_renewal', 'port', 'search_type', 'secondary_server', 'secure', 'server', 'server_identity_check', 'source_ip', 'source_port', 'ssl_min_proto_version', 'tertiary_server', 'two_factor', 'two_factor_authentication', 'two_factor_filter', 'two_factor_notification', 'type', 'user_info_exchange_server', 'username']
    json = remove_invalid_fields(json)
    dictionary = {}
    for attribute in option_list:
        if attribute in json and json[attribute] is not None:
            dictionary[attribute] = json[attribute]
    return dictionary