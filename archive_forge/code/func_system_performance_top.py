from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import Connection
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.data_post_processor import (
def system_performance_top(data, fos):
    vdom = data['vdom']
    system_performance_top_data = data['system_performance_top']
    filtered_data = underscore_to_hyphen(filter_system_performance_top_data(system_performance_top_data))
    converted_data = valid_attr_to_invalid_attrs(filtered_data)
    return fos.set('system.performance', 'top', data=converted_data, vdom=vdom)