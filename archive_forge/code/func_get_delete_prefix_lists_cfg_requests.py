from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts \
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils \
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
def get_delete_prefix_lists_cfg_requests(self, commands, have):
    """Traverse the input list of configuration "delete" commands obtained
        from parsing the input playbook parameters. For each command,
        create and return the appropriate set of REST API requests to delete
        the prefix set configuration specified by the current "command"."""
    requests = []
    for command in commands:
        new_requests = self.get_delete_single_prefix_cfg_requests(command, have)
        if new_requests and len(new_requests) > 0:
            requests.extend(new_requests)
    return requests