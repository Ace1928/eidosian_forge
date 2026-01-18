from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts \
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils \
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
def get_modify_prefix_lists_requests(self, commands):
    """Traverse the input list of configuration "modify" commands obtained
        from parsing the input playbook parameters. For each command,
        create and return the appropriate set of REST API requests to modify
        the prefix set specified by the current command."""
    requests = []
    if not commands:
        return requests
    prefix_set_payload_list = []
    for command in commands:
        prefix_set_payload = self.get_modify_single_prefix_set_request(command)
        if prefix_set_payload:
            prefix_set_payload_list.append(prefix_set_payload)
    prefix_set_data = {self.prefix_set_data_path: prefix_set_payload_list}
    request = {'path': self.prefix_set_uri, 'method': PATCH, 'data': prefix_set_data}
    requests.append(request)
    return requests