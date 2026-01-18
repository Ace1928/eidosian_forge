from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import Connection
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.data_post_processor import (
def hardware_npu_np6_session_stats(data, fos):
    vdom = data['vdom']
    hardware_npu_np6_session_stats_data = data['hardware_npu_np6_session_stats']
    filtered_data = underscore_to_hyphen(filter_hardware_npu_np6_session_stats_data(hardware_npu_np6_session_stats_data))
    converted_data = valid_attr_to_invalid_attrs(filtered_data)
    return fos.set('hardware.npu.np6', 'session-stats', data=converted_data, vdom=vdom)