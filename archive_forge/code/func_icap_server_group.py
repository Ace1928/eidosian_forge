from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import Connection
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.data_post_processor import (
def icap_server_group(data, fos):
    vdom = data['vdom']
    state = data['state']
    icap_server_group_data = data['icap_server_group']
    filtered_data = underscore_to_hyphen(filter_icap_server_group_data(icap_server_group_data))
    if state == 'present' or state is True:
        return fos.set('icap', 'server-group', data=filtered_data, vdom=vdom)
    elif state == 'absent':
        return fos.delete('icap', 'server-group', mkey=filtered_data['name'], vdom=vdom)
    else:
        fos._module.fail_json(msg='state must be present or absent!')