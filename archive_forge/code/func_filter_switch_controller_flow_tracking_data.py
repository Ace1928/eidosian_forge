from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import Connection
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.data_post_processor import (
def filter_switch_controller_flow_tracking_data(json):
    option_list = ['aggregates', 'collector_ip', 'collector_port', 'collectors', 'format', 'level', 'max_export_pkt_size', 'sample_mode', 'sample_rate', 'template_export_period', 'timeout_general', 'timeout_icmp', 'timeout_max', 'timeout_tcp', 'timeout_tcp_fin', 'timeout_tcp_rst', 'timeout_udp', 'transport']
    json = remove_invalid_fields(json)
    dictionary = {}
    for attribute in option_list:
        if attribute in json and json[attribute] is not None:
            dictionary[attribute] = json[attribute]
    return dictionary