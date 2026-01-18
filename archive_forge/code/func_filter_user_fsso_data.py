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
def filter_user_fsso_data(json):
    option_list = ['group_poll_interval', 'interface', 'interface_select_method', 'ldap_poll', 'ldap_poll_filter', 'ldap_poll_interval', 'ldap_server', 'logon_timeout', 'name', 'password', 'password2', 'password3', 'password4', 'password5', 'port', 'port2', 'port3', 'port4', 'port5', 'server', 'server2', 'server3', 'server4', 'server5', 'sni', 'source_ip', 'source_ip6', 'ssl', 'ssl_server_host_ip_check', 'ssl_trusted_cert', 'type', 'user_info_server']
    json = remove_invalid_fields(json)
    dictionary = {}
    for attribute in option_list:
        if attribute in json and json[attribute] is not None:
            dictionary[attribute] = json[attribute]
    return dictionary