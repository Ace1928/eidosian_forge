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
def get_delete_bgp_communities(self, commands, have, is_delete_all):
    requests = []
    if is_delete_all:
        requests = self.get_delete_all_bgp_communities(commands)
    else:
        for cmd in commands:
            name = cmd['name']
            members = cmd.get('members', None)
            cmd_type = cmd['type']
            diff_members = []
            for item in have:
                if item['name'] == name:
                    if 'permit' not in cmd or cmd['permit'] is None:
                        cmd['permit'] = item['permit']
                    if cmd == item:
                        requests.append(self.get_delete_single_bgp_community_requests(name))
                        break
                    if cmd_type == 'standard':
                        for attr in self.standard_communities_map:
                            if cmd.get(attr, None) and item[attr] and (cmd[attr] == item[attr]):
                                diff_members.append(self.standard_communities_map[attr])
                    if members:
                        if members.get('regex', []):
                            for member_want in members['regex']:
                                if item.get('members', None) and item['members'].get('regex', []):
                                    if str(member_want) in item['members']['regex']:
                                        diff_members.append('REGEX:' + str(member_want))
                        else:
                            requests.append(self.get_delete_single_bgp_community_requests(name))
                    elif cmd_type == 'standard':
                        no_attr = True
                        for attr in self.standard_communities_map:
                            if cmd.get(attr, None):
                                no_attr = False
                                break
                        if no_attr:
                            requests.append(self.get_delete_single_bgp_community_requests(name))
                    else:
                        requests.append(self.get_delete_single_bgp_community_requests(name))
                    break
            if diff_members:
                requests.extend(self.get_delete_single_bgp_community_member_requests(name, diff_members))
    return requests