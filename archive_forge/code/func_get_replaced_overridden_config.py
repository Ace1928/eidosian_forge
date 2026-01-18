from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import to_request
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible.module_utils.connection import ConnectionError
import json
from ansible.module_utils._text import to_native
import traceback
def get_replaced_overridden_config(self, want, have, cur_state):
    commands, requests = ([], [])
    commands_del, requests_del = ([], [])
    commands_add, requests_add = ([], [])
    for conf in want:
        name = conf['name']
        in_have = False
        for have_conf in have:
            if have_conf['name'] == name:
                in_have = True
                if have_conf['type'] != conf['type']:
                    commands_del.append(have_conf)
                    commands_add.append(conf)
                else:
                    is_change = False
                    if have_conf['permit'] != conf['permit']:
                        is_change = True
                    if have_conf['match'] != conf['match']:
                        is_change = is_delete = True
                    if conf['type'] == 'standard':
                        no_attr = True
                        for attr in self.standard_communities_map:
                            if not conf.get(attr, None):
                                if have_conf.get(attr, None):
                                    is_change = True
                            else:
                                no_attr = False
                                if not have_conf.get(attr, None):
                                    is_change = True
                        if no_attr:
                            self._module.fail_json(msg='Cannot create standard community-list {0} without community attributes'.format(conf['name']))
                    else:
                        members = conf.get('members', {})
                        if members and members.get('regex', []):
                            if have_conf.get('members', {}) and have_conf['members'].get('regex', []):
                                if set(have_conf['members']['regex']).symmetric_difference(set(members['regex'])):
                                    is_change = True
                        else:
                            self._module.fail_json(msg='Cannot create expanded community-list {0} without community attributes'.format(conf['name']))
                    if is_change:
                        commands_add.append(conf)
                        commands_del.append(have_conf)
                break
        if not in_have:
            commands_add.append(conf)
    if cur_state == 'overridden':
        for have_conf in have:
            in_want = next((conf for conf in want if conf['name'] == have_conf['name']), None)
            if not in_want:
                commands_del.append(have_conf)
    if commands_del:
        requests_del = self.get_delete_bgp_communities(commands_del, have, False)
        if len(requests_del) > 0:
            commands.extend(update_states(commands_del, 'deleted'))
            requests.extend(requests_del)
    if commands_add:
        requests_add = self.get_modify_bgp_community_requests(commands_add, have, cur_state)
        if len(requests_add) > 0:
            commands.extend(update_states(commands_add, cur_state))
            requests.extend(requests_add)
    return (commands, requests)