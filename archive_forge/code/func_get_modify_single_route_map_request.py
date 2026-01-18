from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts \
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils \
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
def get_modify_single_route_map_request(self, command, have):
    """Create and return the appropriate set of route map REST API attributes
        to modify the route map configuration specified by the current "command"."""
    request = {}
    if not command:
        return request
    conf_map_name = command.get('map_name', None)
    conf_action = command.get('action', None)
    conf_seq_num = command.get('sequence_num', None)
    if not conf_map_name or not conf_action or (not conf_seq_num):
        return request
    req_seq_num = str(conf_seq_num)
    if conf_action == 'permit':
        req_action = 'ACCEPT_ROUTE'
    elif conf_action == 'deny':
        req_action = 'REJECT_ROUTE'
    else:
        return request
    route_map_request = {'name': conf_map_name, 'config': {'name': conf_map_name}, 'statements': {'statement': [{'name': req_seq_num, 'config': {'name': req_seq_num}, 'actions': {'config': {'policy-result': req_action}}}]}}
    route_map_statement = route_map_request['statements']['statement'][0]
    self.get_route_map_modify_match_attr(command, route_map_statement)
    self.get_route_map_modify_set_attr(command, route_map_statement, have)
    self.get_route_map_modify_call_attr(command, route_map_statement)
    return route_map_request