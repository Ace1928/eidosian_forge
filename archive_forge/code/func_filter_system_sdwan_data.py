from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import Connection
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.data_post_processor import (
def filter_system_sdwan_data(json):
    option_list = ['app_perf_log_period', 'duplication', 'duplication_max_num', 'fail_alert_interfaces', 'fail_detect', 'health_check', 'load_balance_mode', 'members', 'neighbor', 'neighbor_hold_boot_time', 'neighbor_hold_down', 'neighbor_hold_down_time', 'service', 'speedtest_bypass_routing', 'status', 'zone']
    json = remove_invalid_fields(json)
    dictionary = {}
    for attribute in option_list:
        if attribute in json and json[attribute] is not None:
            dictionary[attribute] = json[attribute]
    return dictionary