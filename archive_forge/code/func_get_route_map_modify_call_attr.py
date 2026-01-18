from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts \
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils \
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
@staticmethod
def get_route_map_modify_call_attr(command, route_map_statement):
    """In the dict specified by the input route_map_statement paramenter,
        provide REST API definitions of the "call" attribute (if present)
        contained in the user input command dict specified by the "command"
        input parameter to this function."""
    call_val = command.get('call')
    if not call_val:
        return
    if not route_map_statement.get('conditions'):
        route_map_statement['conditions'] = {'config': {}}
    elif not route_map_statement['conditions'].get('config'):
        route_map_statement['conditions']['config'] = {}
    route_map_statement['conditions']['config']['call-policy'] = call_val