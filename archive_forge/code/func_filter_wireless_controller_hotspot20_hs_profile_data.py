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
def filter_wireless_controller_hotspot20_hs_profile_data(json):
    option_list = ['plmn_3gpp', 'access_network_asra', 'access_network_esr', 'access_network_internet', 'access_network_type', 'access_network_uesa', 'advice_of_charge', 'anqp_domain_id', 'bss_transition', 'conn_cap', 'deauth_request_timeout', 'dgaf', 'domain_name', 'gas_comeback_delay', 'gas_fragmentation_limit', 'hessid', 'ip_addr_type', 'l2tif', 'nai_realm', 'name', 'network_auth', 'oper_friendly_name', 'oper_icon', 'osu_provider', 'osu_provider_nai', 'osu_ssid', 'pame_bi', 'proxy_arp', 'qos_map', 'release', 'roaming_consortium', 'terms_and_conditions', 'venue_group', 'venue_name', 'venue_type', 'venue_url', 'wan_metrics', 'wnm_sleep_mode']
    json = remove_invalid_fields(json)
    dictionary = {}
    for attribute in option_list:
        if attribute in json and json[attribute] is not None:
            dictionary[attribute] = json[attribute]
    return dictionary