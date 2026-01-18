from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import Connection
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.data_post_processor import (
def filter_log_disk_filter_data(json):
    option_list = ['admin', 'anomaly', 'auth', 'cpu_memory_usage', 'dhcp', 'dlp_archive', 'dns', 'event', 'filter', 'filter_type', 'forward_traffic', 'free_style', 'gtp', 'ha', 'ipsec', 'ldb_monitor', 'local_traffic', 'multicast_traffic', 'netscan_discovery', 'netscan_vulnerability', 'notification', 'pattern', 'ppp', 'radius', 'severity', 'sniffer_traffic', 'ssh', 'sslvpn_log_adm', 'sslvpn_log_auth', 'sslvpn_log_session', 'system', 'vip_ssl', 'voip', 'wan_opt', 'wireless_activity', 'ztna_traffic']
    json = remove_invalid_fields(json)
    dictionary = {}
    for attribute in option_list:
        if attribute in json and json[attribute] is not None:
            dictionary[attribute] = json[attribute]
    return dictionary