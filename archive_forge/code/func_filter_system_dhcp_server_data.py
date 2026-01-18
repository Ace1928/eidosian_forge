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
def filter_system_dhcp_server_data(json):
    option_list = ['auto_configuration', 'auto_managed_status', 'conflicted_ip_timeout', 'ddns_auth', 'ddns_key', 'ddns_keyname', 'ddns_server_ip', 'ddns_ttl', 'ddns_update', 'ddns_update_override', 'ddns_zone', 'default_gateway', 'dhcp_settings_from_fortiipam', 'dns_server1', 'dns_server2', 'dns_server3', 'dns_server4', 'dns_service', 'domain', 'exclude_range', 'filename', 'forticlient_on_net_status', 'id', 'interface', 'ip_mode', 'ip_range', 'ipsec_lease_hold', 'lease_time', 'mac_acl_default_action', 'netmask', 'next_server', 'ntp_server1', 'ntp_server2', 'ntp_server3', 'ntp_service', 'options', 'relay_agent', 'reserved_address', 'server_type', 'shared_subnet', 'status', 'tftp_server', 'timezone', 'timezone_option', 'vci_match', 'vci_string', 'wifi_ac_service', 'wifi_ac1', 'wifi_ac2', 'wifi_ac3', 'wins_server1', 'wins_server2']
    json = remove_invalid_fields(json)
    dictionary = {}
    for attribute in option_list:
        if attribute in json and json[attribute] is not None:
            dictionary[attribute] = json[attribute]
    return dictionary