from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import Connection
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.data_post_processor import (
def switch_controller_poe(data, fos):
    vdom = data['vdom']
    switch_controller_poe_data = data['switch_controller_poe']
    filtered_data = underscore_to_hyphen(filter_switch_controller_poe_data(switch_controller_poe_data))
    converted_data = valid_attr_to_invalid_attrs(filtered_data)
    return fos.set('switch-controller', 'poe', data=converted_data, vdom=vdom)