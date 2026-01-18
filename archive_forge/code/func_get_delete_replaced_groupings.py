from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts \
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils \
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
def get_delete_replaced_groupings(self, commands, have):
    """For each of the route maps specified in the "commands" input list,
        create requests to delete any existing route map configuration
        groupings for which modified attribute requests are specified."""
    requests = []
    for command in commands:
        self.get_delete_one_map_replaced_groupings(command, have, requests)
    return requests