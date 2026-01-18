from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.six.moves.urllib.parse import urlencode
from ansible_collections.community.hrobot.plugins.module_utils.robot import (
def set_desired_servers(module, id_):
    v_switch = get_v_switch(module, id_)
    changed = False
    if module.params['servers'] is None:
        return (v_switch, changed)
    servers_to_delete = get_servers_to_delete(v_switch['server'], module.params['servers'])
    if servers_to_delete:
        if not module.check_mode:
            v_switch = delete_servers(module, id_, servers_to_delete)
        changed = True
    servers_to_add = get_servers_to_add(v_switch['server'], module.params['servers'])
    if servers_to_add:
        if not module.check_mode:
            v_switch = add_servers(module, id_, servers_to_add)
        changed = True
    return (v_switch, changed)