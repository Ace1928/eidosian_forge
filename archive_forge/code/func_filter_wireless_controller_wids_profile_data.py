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
def filter_wireless_controller_wids_profile_data(json):
    option_list = ['ap_auto_suppress', 'ap_bgscan_disable_day', 'ap_bgscan_disable_end', 'ap_bgscan_disable_schedules', 'ap_bgscan_disable_start', 'ap_bgscan_duration', 'ap_bgscan_idle', 'ap_bgscan_intv', 'ap_bgscan_period', 'ap_bgscan_report_intv', 'ap_fgscan_report_intv', 'ap_scan', 'ap_scan_channel_list_2G_5G', 'ap_scan_channel_list_6G', 'ap_scan_passive', 'ap_scan_threshold', 'asleap_attack', 'assoc_flood_thresh', 'assoc_flood_time', 'assoc_frame_flood', 'auth_flood_thresh', 'auth_flood_time', 'auth_frame_flood', 'comment', 'deauth_broadcast', 'deauth_unknown_src_thresh', 'eapol_fail_flood', 'eapol_fail_intv', 'eapol_fail_thresh', 'eapol_logoff_flood', 'eapol_logoff_intv', 'eapol_logoff_thresh', 'eapol_pre_fail_flood', 'eapol_pre_fail_intv', 'eapol_pre_fail_thresh', 'eapol_pre_succ_flood', 'eapol_pre_succ_intv', 'eapol_pre_succ_thresh', 'eapol_start_flood', 'eapol_start_intv', 'eapol_start_thresh', 'eapol_succ_flood', 'eapol_succ_intv', 'eapol_succ_thresh', 'invalid_mac_oui', 'long_duration_attack', 'long_duration_thresh', 'name', 'null_ssid_probe_resp', 'sensor_mode', 'spoofed_deauth', 'weak_wep_iv', 'wireless_bridge']
    json = remove_invalid_fields(json)
    dictionary = {}
    for attribute in option_list:
        if attribute in json and json[attribute] is not None:
            dictionary[attribute] = json[attribute]
    return dictionary