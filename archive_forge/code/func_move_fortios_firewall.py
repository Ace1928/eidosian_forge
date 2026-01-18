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
def move_fortios_firewall(data, fos):
    if not data['self'] or (not data['after'] and (not data['before'])):
        fos._module.fail_json(msg='self, after(or before) must not be empty')
    vdom = data['vdom']
    params_set = dict()
    params_set['action'] = 'move'
    if data['after']:
        params_set['after'] = data['after']
    if data['before']:
        params_set['before'] = data['before']
    return fos.set('firewall', 'shaping-policy', data=None, mkey=data['self'], vdom=vdom, parameters=params_set)