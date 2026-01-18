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
def filter_wireless_controller_wtp_data(json):
    option_list = ['admin', 'allowaccess', 'apcfg_profile', 'ble_major_id', 'ble_minor_id', 'bonjour_profile', 'coordinate_enable', 'coordinate_latitude', 'coordinate_longitude', 'coordinate_x', 'coordinate_y', 'firmware_provision', 'firmware_provision_latest', 'image_download', 'index', 'ip_fragment_preventing', 'lan', 'led_state', 'location', 'login_passwd', 'login_passwd_change', 'mesh_bridge_enable', 'name', 'override_allowaccess', 'override_ip_fragment', 'override_lan', 'override_led_state', 'override_login_passwd_change', 'override_split_tunnel', 'override_wan_port_mode', 'radio_1', 'radio_2', 'radio_3', 'radio_4', 'region', 'region_x', 'region_y', 'split_tunneling_acl', 'split_tunneling_acl_local_ap_subnet', 'split_tunneling_acl_path', 'tun_mtu_downlink', 'tun_mtu_uplink', 'uuid', 'wan_port_mode', 'wtp_id', 'wtp_mode', 'wtp_profile']
    json = remove_invalid_fields(json)
    dictionary = {}
    for attribute in option_list:
        if attribute in json and json[attribute] is not None:
            dictionary[attribute] = json[attribute]
    return dictionary