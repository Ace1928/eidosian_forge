from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible.module_utils.connection import ConnectionError
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
def get_delete_port_breakout_requests(self, commands, have):
    requests = []
    if not commands:
        return requests
    have_new = self.get_all_breakout_mode(have)
    for conf in commands:
        name = conf['name']
        match = next((cfg for cfg in have_new if cfg['name'] == name), None)
        req = self.get_delete_single_port_breakout(name, match)
        if req:
            requests.append(req)
    return requests