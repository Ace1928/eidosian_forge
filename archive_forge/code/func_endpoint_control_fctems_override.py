from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import Connection
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.data_post_processor import (
def endpoint_control_fctems_override(data, fos):
    vdom = data['vdom']
    state = data['state']
    endpoint_control_fctems_override_data = data['endpoint_control_fctems_override']
    endpoint_control_fctems_override_data = flatten_multilists_attributes(endpoint_control_fctems_override_data)
    filtered_data = underscore_to_hyphen(filter_endpoint_control_fctems_override_data(endpoint_control_fctems_override_data))
    if state == 'present' or state is True:
        return fos.set('endpoint-control', 'fctems-override', data=filtered_data, vdom=vdom)
    elif state == 'absent':
        return fos.delete('endpoint-control', 'fctems-override', mkey=filtered_data['ems-id'], vdom=vdom)
    else:
        fos._module.fail_json(msg='state must be present or absent!')