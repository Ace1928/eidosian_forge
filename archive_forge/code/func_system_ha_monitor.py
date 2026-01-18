from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import Connection
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.data_post_processor import (
def system_ha_monitor(data, fos):
    vdom = data['vdom']
    system_ha_monitor_data = data['system_ha_monitor']
    filtered_data = underscore_to_hyphen(filter_system_ha_monitor_data(system_ha_monitor_data))
    return fos.set('system', 'ha-monitor', data=filtered_data, vdom=vdom)