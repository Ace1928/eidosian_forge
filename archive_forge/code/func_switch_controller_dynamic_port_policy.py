from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import Connection
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.data_post_processor import (
def switch_controller_dynamic_port_policy(data, fos):
    vdom = data['vdom']
    state = data['state']
    switch_controller_dynamic_port_policy_data = data['switch_controller_dynamic_port_policy']
    filtered_data = underscore_to_hyphen(filter_switch_controller_dynamic_port_policy_data(switch_controller_dynamic_port_policy_data))
    converted_data = valid_attr_to_invalid_attrs(filtered_data)
    if state == 'present' or state is True:
        return fos.set('switch-controller', 'dynamic-port-policy', data=converted_data, vdom=vdom)
    elif state == 'absent':
        return fos.delete('switch-controller', 'dynamic-port-policy', mkey=filtered_data['name'], vdom=vdom)
    else:
        fos._module.fail_json(msg='state must be present or absent!')