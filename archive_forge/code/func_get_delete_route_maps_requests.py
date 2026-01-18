from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts \
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils \
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
def get_delete_route_maps_requests(self, have, commands):
    """Traverse the input list of configuration "delete" commands obtained
        from parsing the input playbook parameters. For each command,
        create and return the appropriate set of REST API requests to delete
        the appropriate elements from the route map specified by the current command."""
    requests = []
    if commands:
        for command in commands:
            self.get_delete_single_route_map_requests(have, command, requests)
    return requests