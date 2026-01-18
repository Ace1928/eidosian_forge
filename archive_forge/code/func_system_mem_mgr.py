from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import Connection
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.data_post_processor import (
def system_mem_mgr(data, fos):
    vdom = data['vdom']
    system_mem_mgr_data = data['system_mem_mgr']
    filtered_data = underscore_to_hyphen(filter_system_mem_mgr_data(system_mem_mgr_data))
    return fos.set('system', 'mem-mgr', data=filtered_data, vdom=vdom)