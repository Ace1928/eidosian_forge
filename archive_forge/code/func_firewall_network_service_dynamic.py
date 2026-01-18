from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import Connection
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.data_post_processor import (
def firewall_network_service_dynamic(data, fos):
    vdom = data['vdom']
    state = data['state']
    firewall_network_service_dynamic_data = data['firewall_network_service_dynamic']
    filtered_data = underscore_to_hyphen(filter_firewall_network_service_dynamic_data(firewall_network_service_dynamic_data))
    if state == 'present' or state is True:
        return fos.set('firewall', 'network-service-dynamic', data=filtered_data, vdom=vdom)
    elif state == 'absent':
        return fos.delete('firewall', 'network-service-dynamic', mkey=filtered_data['name'], vdom=vdom)
    else:
        fos._module.fail_json(msg='state must be present or absent!')