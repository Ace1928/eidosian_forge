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
def filter_wireless_controller_qos_profile_data(json):
    option_list = ['bandwidth_admission_control', 'bandwidth_capacity', 'burst', 'call_admission_control', 'call_capacity', 'comment', 'downlink', 'downlink_sta', 'dscp_wmm_be', 'dscp_wmm_bk', 'dscp_wmm_mapping', 'dscp_wmm_vi', 'dscp_wmm_vo', 'name', 'uplink', 'uplink_sta', 'wmm', 'wmm_be_dscp', 'wmm_bk_dscp', 'wmm_dscp_marking', 'wmm_uapsd', 'wmm_vi_dscp', 'wmm_vo_dscp']
    json = remove_invalid_fields(json)
    dictionary = {}
    for attribute in option_list:
        if attribute in json and json[attribute] is not None:
            dictionary[attribute] = json[attribute]
    return dictionary