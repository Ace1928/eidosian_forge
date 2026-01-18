from __future__ import absolute_import, division, print_function
import json
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.formatted_diff_utils import (
from ansible.module_utils.connection import ConnectionError
def get_modify_users_requests(self, commands, have):
    requests = []
    if not commands:
        return requests
    for conf in commands:
        match = next((cfg for cfg in have if cfg['name'] == conf['name']), None)
        req = self.get_modify_single_user_request(conf, match)
        if req:
            requests.append(req)
    return requests