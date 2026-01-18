from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import Connection
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.data_post_processor import (
def system_management_tunnel(data, fos):
    vdom = data['vdom']
    system_management_tunnel_data = data['system_management_tunnel']
    filtered_data = underscore_to_hyphen(filter_system_management_tunnel_data(system_management_tunnel_data))
    return fos.set('system', 'management-tunnel', data=filtered_data, vdom=vdom)