from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import Connection
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.data_post_processor import (
def system_replacemsg_automation(data, fos):
    vdom = data['vdom']
    state = data['state']
    system_replacemsg_automation_data = data['system_replacemsg_automation']
    filtered_data = underscore_to_hyphen(filter_system_replacemsg_automation_data(system_replacemsg_automation_data))
    if state == 'present' or state is True:
        return fos.set('system.replacemsg', 'automation', data=filtered_data, vdom=vdom)
    elif state == 'absent':
        return fos.delete('system.replacemsg', 'automation', mkey=filtered_data['msg-type'], vdom=vdom)
    else:
        fos._module.fail_json(msg='state must be present or absent!')