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
def filter_wireless_controller_wtp_profile_data(json):
    option_list = ['allowaccess', 'ap_country', 'ap_handoff', 'apcfg_profile', 'ble_profile', 'comment', 'console_login', 'control_message_offload', 'deny_mac_list', 'dtls_in_kernel', 'dtls_policy', 'energy_efficient_ethernet', 'esl_ses_dongle', 'ext_info_enable', 'frequency_handoff', 'handoff_roaming', 'handoff_rssi', 'handoff_sta_thresh', 'indoor_outdoor_deployment', 'ip_fragment_preventing', 'lan', 'lbs', 'led_schedules', 'led_state', 'lldp', 'login_passwd', 'login_passwd_change', 'max_clients', 'name', 'platform', 'poe_mode', 'radio_1', 'radio_2', 'radio_3', 'radio_4', 'split_tunneling_acl', 'split_tunneling_acl_local_ap_subnet', 'split_tunneling_acl_path', 'syslog_profile', 'tun_mtu_downlink', 'tun_mtu_uplink', 'unii_4_5ghz_band', 'wan_port_auth', 'wan_port_auth_methods', 'wan_port_auth_password', 'wan_port_auth_usrname', 'wan_port_mode']
    json = remove_invalid_fields(json)
    dictionary = {}
    for attribute in option_list:
        if attribute in json and json[attribute] is not None:
            dictionary[attribute] = json[attribute]
    return dictionary