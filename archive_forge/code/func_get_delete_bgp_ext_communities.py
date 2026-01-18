from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import to_request
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
import json
from ansible.module_utils._text import to_native
from ansible.module_utils.connection import ConnectionError
import traceback
def get_delete_bgp_ext_communities(self, commands, have, is_delete_all):
    requests = []
    if is_delete_all:
        requests = self.get_delete_all_bgp_ext_communities(commands)
    else:
        for cmd in commands:
            name = cmd['name']
            cmd_type = cmd['type']
            members = cmd.get('members', None)
            diff_members = []
            for item in have:
                if item['name'] == name:
                    if 'permit' not in cmd or cmd['permit'] is None:
                        cmd['permit'] = item['permit']
                    if cmd == item:
                        requests.append(self.get_delete_single_bgp_ext_community_requests(name))
                        break
                    if members:
                        if cmd_type == 'expanded':
                            if members.get('regex', []):
                                for member_want in members['regex']:
                                    if item.get('members', None) and item['members'].get('regex', []):
                                        if str(member_want) in item['members']['regex']:
                                            diff_members.append('REGEX:' + str(member_want))
                            else:
                                requests.append(self.get_delete_single_bgp_ext_community_requests(name))
                        else:
                            no_members = True
                            for attr in self.standard_communities_map:
                                if members.get(attr, []):
                                    no_members = False
                                    for member_want in members[attr]:
                                        if item.get('members', None) and item['members'].get(attr, []):
                                            if str(member_want) in item['members'][attr]:
                                                diff_members.append(self.standard_communities_map[attr] + ':' + str(member_want))
                            if no_members:
                                requests.append(self.get_delete_single_bgp_ext_community_requests(name))
                    else:
                        requests.append(self.get_delete_single_bgp_ext_community_requests(name))
                    break
            if diff_members:
                requests.extend(self.get_delete_single_bgp_ext_community_member_requests(name, diff_members))
    return requests