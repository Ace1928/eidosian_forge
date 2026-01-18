from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.facts.facts import Facts
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.utils.utils import (
def del_intf_commands(self, w, have):
    commands = []
    obj_in_have = search_obj_in_list(w['name'], have, 'name')
    if obj_in_have:
        lst_to_del = self.intersect_list_of_dicts(w['members'], obj_in_have['members'])
        if lst_to_del:
            for item in lst_to_del:
                commands.append('interface' + ' ' + item['member'])
                commands.append('no channel-group')
    return commands