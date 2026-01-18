from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import Connection
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.data_post_processor import (
def filter_system_modem_data(json):
    option_list = ['action', 'altmode', 'authtype1', 'authtype2', 'authtype3', 'auto_dial', 'connect_timeout', 'dial_cmd1', 'dial_cmd2', 'dial_cmd3', 'dial_on_demand', 'distance', 'dont_send_CR1', 'dont_send_CR2', 'dont_send_CR3', 'extra_init1', 'extra_init2', 'extra_init3', 'holddown_timer', 'idle_timer', 'interface', 'lockdown_lac', 'mode', 'network_init', 'passwd1', 'passwd2', 'passwd3', 'peer_modem1', 'peer_modem2', 'peer_modem3', 'phone1', 'phone2', 'phone3', 'pin_init', 'ppp_echo_request1', 'ppp_echo_request2', 'ppp_echo_request3', 'priority', 'redial', 'reset', 'status', 'traffic_check', 'username1', 'username2', 'username3', 'wireless_port']
    json = remove_invalid_fields(json)
    dictionary = {}
    for attribute in option_list:
        if attribute in json and json[attribute] is not None:
            dictionary[attribute] = json[attribute]
    return dictionary