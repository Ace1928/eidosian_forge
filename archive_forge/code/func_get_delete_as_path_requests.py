from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible.module_utils.connection import ConnectionError
def get_delete_as_path_requests(self, commands, have, is_delete_all):
    requests = []
    if is_delete_all:
        requests = self.get_delete_all_as_path_requests(commands)
    else:
        for cmd in commands:
            name = cmd['name']
            members = cmd['members']
            permit = cmd['permit']
            match = next((item for item in have if item['name'] == cmd['name']), None)
            if match:
                if members:
                    if match.get('members'):
                        del_members = set(match['members']).intersection(set(members))
                        if del_members:
                            if len(del_members) == len(match['members']):
                                requests.append(self.get_delete_single_as_path_request(name))
                            else:
                                requests.append(self.get_delete_single_as_path_member_request(name, del_members))
                else:
                    requests.append(self.get_delete_single_as_path_request(name))
    return requests