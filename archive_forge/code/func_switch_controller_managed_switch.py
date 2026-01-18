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
def switch_controller_managed_switch(data, fos, check_mode=False):
    vdom = data['vdom']
    state = data['state']
    switch_controller_managed_switch_data = data['switch_controller_managed_switch']
    switch_controller_managed_switch_data = flatten_multilists_attributes(switch_controller_managed_switch_data)
    filtered_data = underscore_to_hyphen(filter_switch_controller_managed_switch_data(switch_controller_managed_switch_data))
    converted_data = valid_attr_to_invalid_attrs(filtered_data)
    if check_mode:
        diff = {'before': '', 'after': filtered_data}
        mkey = fos.get_mkey('switch-controller', 'managed-switch', filtered_data, vdom=vdom)
        current_data = fos.get('switch-controller', 'managed-switch', vdom=vdom, mkey=mkey)
        is_existed = current_data and current_data.get('http_status') == 200 and isinstance(current_data.get('results'), list) and (len(current_data['results']) > 0)
        if state == 'present' or state is True:
            if mkey is None:
                return (False, True, filtered_data, diff)
            if is_existed:
                is_same = is_same_comparison(serialize(current_data['results'][0]), serialize(filtered_data))
                current_values = find_current_values(current_data['results'][0], filtered_data)
                return (False, not is_same, filtered_data, {'before': current_values, 'after': filtered_data})
            return (False, True, filtered_data, diff)
        if state == 'absent':
            if mkey is None:
                return (False, False, filtered_data, {'before': current_data['results'][0], 'after': ''})
            if is_existed:
                return (False, True, filtered_data, {'before': current_data['results'][0], 'after': ''})
            return (False, False, filtered_data, {})
        return (True, False, {'reason: ': 'Must provide state parameter'}, {})
    if state == 'present' or state is True:
        return fos.set('switch-controller', 'managed-switch', data=converted_data, vdom=vdom)
    elif state == 'absent':
        return fos.delete('switch-controller', 'managed-switch', mkey=filtered_data['switch-id'], vdom=vdom)
    else:
        fos._module.fail_json(msg='state must be present or absent!')