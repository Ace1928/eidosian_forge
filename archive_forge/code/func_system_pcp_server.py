from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import Connection
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.data_post_processor import (
def system_pcp_server(data, fos):
    vdom = data['vdom']
    system_pcp_server_data = data['system_pcp_server']
    system_pcp_server_data = flatten_multilists_attributes(system_pcp_server_data)
    filtered_data = underscore_to_hyphen(filter_system_pcp_server_data(system_pcp_server_data))
    return fos.set('system', 'pcp-server', data=filtered_data, vdom=vdom)