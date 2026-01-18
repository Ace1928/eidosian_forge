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
def template_for_lag_creation(self, have, diff_members, diff_portchannels, state_name):
    commands = list()
    requests = list()
    if diff_members:
        commands_portchannels, requests = self.call_create_port_channel(diff_members, have)
        if commands_portchannels:
            po_list = [{'name': x['name']} for x in commands_portchannels if x['name']]
        else:
            po_list = []
        if po_list:
            commands.extend(update_states(po_list, state_name))
        diff_members_remove_none = [x for x in diff_members if x['members']]
        if diff_members_remove_none:
            request = self.create_lag_interfaces_requests(diff_members_remove_none)
            if request:
                requests.extend(request)
            else:
                requests = request
        commands.extend(update_states(diff_members, state_name))
    if diff_portchannels:
        portchannels, po_requests = self.call_create_port_channel(diff_portchannels, have)
        requests.extend(po_requests)
        commands.extend(update_states(portchannels, state_name))
    return (commands, requests)