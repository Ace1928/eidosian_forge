from __future__ import absolute_import, division, print_function
import json
from copy import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.formatted_diff_utils import (
from ansible.module_utils._text import to_native
from ansible.module_utils.connection import ConnectionError
import traceback
def template_for_lag_deletion(self, have, delete_members, delete_portchannels, state_name):
    commands = list()
    requests = list()
    portchannel_requests = list()
    if delete_members:
        delete_members_remove_none = [x for x in delete_members if x['members']]
        requests = self.get_delete_lag_interfaces_requests(delete_members_remove_none)
        delete_all_members = [x for x in delete_members if 'members' in x.keys() and (not x['members'])]
        delete_all_list = list()
        if delete_all_members:
            for i in delete_all_members:
                list_obj = search_obj_in_list(i['name'], have, 'name')
                if list_obj['members']:
                    delete_all_list.append(list_obj)
        if delete_all_list:
            deleteall_requests = self.get_delete_lag_interfaces_requests(delete_all_list)
        else:
            deleteall_requests = []
        if requests and deleteall_requests:
            requests.extend(deleteall_requests)
        elif deleteall_requests:
            requests = deleteall_requests
        if requests:
            commands.extend(update_states(delete_members, state_name))
    if delete_portchannels:
        portchannel_requests = self.get_delete_portchannel_requests(delete_portchannels)
        commands_del = self.prune_commands(delete_portchannels)
        commands.extend(update_states(commands_del, state_name))
    if requests:
        requests.extend(portchannel_requests)
    else:
        requests = portchannel_requests
    return (commands, requests)