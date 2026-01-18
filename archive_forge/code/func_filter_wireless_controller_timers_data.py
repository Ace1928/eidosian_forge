from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import Connection
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.data_post_processor import (
def filter_wireless_controller_timers_data(json):
    option_list = ['auth_timeout', 'ble_scan_report_intv', 'client_idle_rehome_timeout', 'client_idle_timeout', 'darrp_day', 'darrp_optimize', 'darrp_time', 'discovery_interval', 'drma_interval', 'echo_interval', 'fake_ap_log', 'ipsec_intf_cleanup', 'radio_stats_interval', 'rogue_ap_cleanup', 'rogue_ap_log', 'sta_capability_interval', 'sta_locate_timer', 'sta_stats_interval', 'vap_stats_interval']
    json = remove_invalid_fields(json)
    dictionary = {}
    for attribute in option_list:
        if attribute in json and json[attribute] is not None:
            dictionary[attribute] = json[attribute]
    return dictionary