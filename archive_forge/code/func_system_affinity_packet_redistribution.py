from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import Connection
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.data_post_processor import (
def system_affinity_packet_redistribution(data, fos):
    vdom = data['vdom']
    state = data['state']
    system_affinity_packet_redistribution_data = data['system_affinity_packet_redistribution']
    filtered_data = underscore_to_hyphen(filter_system_affinity_packet_redistribution_data(system_affinity_packet_redistribution_data))
    if state == 'present' or state is True:
        return fos.set('system', 'affinity-packet-redistribution', data=filtered_data, vdom=vdom)
    elif state == 'absent':
        return fos.delete('system', 'affinity-packet-redistribution', mkey=filtered_data['id'], vdom=vdom)
    else:
        fos._module.fail_json(msg='state must be present or absent!')